"""
evaluate.py
--------------------------------------------------
Diffusion 기반 딥보이스 탐지 평가 코드

역할:
1. 학습된 모델 체크포인트 불러오기
2. 검증 데이터로 threshold 결정
3. 테스트 데이터의 Mel-spectrogram 복원
4. 원본과 복원본의 차이를 이상 점수로 계산
5. REAL / FAKE 예측
6. Accuracy, Precision, Recall, F1 및 혼동 행렬 계산
7. 테스트 결과를 CSV로 저장

라벨 규칙:
    0 = REAL / bonafide
    1 = FAKE / spoof

판별 규칙:
    score >= threshold  -> FAKE (1)
    score <  threshold  -> REAL (0)

주의:
    threshold는 검증 데이터로만 결정한다.
    테스트 데이터로 threshold를 조정하지 않는다.
"""

from pathlib import Path
import csv

import torch

# 모델 구조는 model.py에서 가져온다.
from model import DiffusionModel

# 학습과 평가에서 공통으로 사용할 설정을 가져온다.
from config import (
    TIMESTEPS,
    IN_CHANNELS,
    HIDDEN_CHANNELS,
    TIME_DIM,
)


# ============================================================
# 1. 실행 장치 설정
# ============================================================
def get_device():
    """GPU 사용이 가능하면 GPU, 아니면 CPU를 사용한다."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# 2. DataLoader batch에서 Mel과 label 추출
# ============================================================
def extract_batch(batch):
    """
    DataLoader에서 Mel-spectrogram과 라벨을 추출한다.

    지원 형식:
        (mel, label)
        (mel, label, path)
        {"mel": mel, "label": label}
        {"x": mel, "label": label}

    Mel shape:
        [B, mel_bins, time_frames]
        또는
        [B, 1, mel_bins, time_frames]
    """

    if isinstance(batch, dict):
        if "mel" in batch:
            mel = batch["mel"]
        elif "x" in batch:
            mel = batch["x"]
        elif "features" in batch:
            mel = batch["features"]
        else:
            raise KeyError(
                "batch에서 Mel 데이터를 찾을 수 없습니다."
            )

        if "label" not in batch:
            raise KeyError(
                "batch에 label이 없습니다."
            )

        labels = batch["label"]

    elif isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError(
                "batch에는 Mel 데이터와 label이 필요합니다."
            )

        mel = batch[0]
        labels = batch[1]

    else:
        raise TypeError(
            f"지원하지 않는 batch 형식: {type(batch)}"
        )

    # Mel 데이터가 Tensor가 아니라면 Tensor로 변환
    if not isinstance(mel, torch.Tensor):
        mel = torch.as_tensor(mel)

    # [B, mel_bins, time_frames] -> [B, 1, mel_bins, time_frames]
    if mel.ndim == 3:
        mel = mel.unsqueeze(1)

    if mel.ndim != 4:
        raise ValueError(
            f"잘못된 Mel shape: {tuple(mel.shape)}"
        )

    mel = mel.float()

    # 라벨을 1차원 정수 Tensor로 변환
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)

    labels = labels.reshape(-1).long()

    # 현재 프로젝트의 라벨 규칙은 REAL=0, FAKE=1
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError(
            "라벨은 REAL=0, FAKE=1이어야 합니다."
        )

    return mel, labels


# ============================================================
# 3. 학습된 모델 불러오기
# ============================================================
def load_model(
    checkpoint_path,
    device=None,
    in_channels=IN_CHANNELS,
    hidden_channels=HIDDEN_CHANNELS,
    time_dim=TIME_DIM,
    timesteps=TIMESTEPS,
):
    """
    train.py에서 저장한 체크포인트를 불러온다.

    주의:
        학습할 때 사용한 모델 구조와 설정이 같아야 한다.
    """

    if device is None:
        device = get_device()

    device = torch.device(device)

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"체크포인트를 찾을 수 없습니다: {checkpoint_path}"
        )

    # 학습 코드에서 저장한 모델과 동일한 구조 생성
    model = DiffusionModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        time_dim=time_dim,
        timesteps=timesteps,
    )

    # CPU에서도 체크포인트를 읽을 수 있도록 map_location 지정
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # train.py의 저장 형식에 맞춰 가중치를 불러온다.
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # state_dict만 직접 저장한 경우도 지원
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    print(f"모델 로드 완료: {checkpoint_path}")
    print(f"사용 장치: {device}")

    return model


# ============================================================
# 4. 데이터셋에서 이상 점수 계산
# ============================================================
@torch.no_grad()
def calculate_scores(
    model,
    data_loader,
    device=None,
    start_t=50,
):
    """
    각 샘플의 이상 점수와 실제 라벨을 계산한다.

    이상 점수:
        원본 Mel과 복원 Mel 사이의 평균 절대 오차

    score가 클수록 원본과 복원 결과의 차이가 크다.

    반환:
        scores: 이상 점수 목록
        labels: 실제 라벨 목록
    """

    if device is None:
        device = next(model.parameters()).device

    device = torch.device(device)

    model.eval()

    all_scores = []
    all_labels = []

    for batch_index, batch in enumerate(data_loader, start=1):

        # ----------------------------------------------------
        # 1) Mel과 실제 라벨 가져오기
        # ----------------------------------------------------
        x_0, labels = extract_batch(batch)

        x_0 = x_0.to(device)
        labels = labels.to(device)

        # ----------------------------------------------------
        # 2) 원본 Mel을 노이즈화한 뒤 복원
        # ----------------------------------------------------
        # model.reconstruct():
        #   원본 -> 노이즈 추가 -> 역방향 Diffusion
        x_reconstructed = model.reconstruct(
            x_0,
            start_t=start_t,
        )

        # ----------------------------------------------------
        # 3) 원본과 복원본의 차이 계산
        # ----------------------------------------------------
        # 샘플마다 평균 절대 오차를 하나의 점수로 만든다.
        difference = torch.abs(
            x_0 - x_reconstructed
        )

        scores = difference.flatten(
            start_dim=1
        ).mean(dim=1)

        # ----------------------------------------------------
        # 4) CPU로 옮겨 저장
        # ----------------------------------------------------
        all_scores.extend(
            scores.detach().cpu().tolist()
        )

        all_labels.extend(
            labels.detach().cpu().tolist()
        )

        if batch_index % 20 == 0:
            print(
                f"평가 진행: {batch_index}/{len(data_loader)}"
            )

    return all_scores, all_labels


# ============================================================
# 5. Threshold 적용
# ============================================================
def predict_labels(scores, threshold):
    """
    이상 점수를 REAL/FAKE 라벨로 변환한다.

    score >= threshold -> FAKE (1)
    score < threshold  -> REAL (0)
    """

    return [
        1 if score >= threshold else 0
        for score in scores
    ]


# ============================================================
# 6. 평가 지표 계산
# ============================================================
def calculate_metrics(labels, predictions):
    """
    REAL=0, FAKE=1 기준으로 평가 지표를 계산한다.

    혼동 행렬:
        [[TN, FP],
         [FN, TP]]

    TN: REAL을 REAL로 예측
    FP: REAL을 FAKE로 잘못 예측
    FN: FAKE를 REAL로 잘못 예측
    TP: FAKE를 FAKE로 예측
    """

    if len(labels) != len(predictions):
        raise ValueError(
            "실제 라벨과 예측 라벨의 길이가 다릅니다."
        )

    if len(labels) == 0:
        raise ValueError("평가할 데이터가 없습니다.")

    tn = sum(
        y == 0 and p == 0
        for y, p in zip(labels, predictions)
    )

    fp = sum(
        y == 0 and p == 1
        for y, p in zip(labels, predictions)
    )

    fn = sum(
        y == 1 and p == 0
        for y, p in zip(labels, predictions)
    )

    tp = sum(
        y == 1 and p == 1
        for y, p in zip(labels, predictions)
    )

    total = len(labels)

    accuracy = (tp + tn) / total

    # FAKE(1) 기준 Precision
    precision = tp / max(tp + fp, 1)

    # FAKE(1) 기준 Recall
    recall = tp / max(tp + fn, 1)

    # FAKE(1) 기준 F1
    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    return {
        "accuracy": accuracy,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total": total,
    }


# ============================================================
# 7. 검증 데이터로 Threshold 선택
# ============================================================
def find_best_threshold(scores, labels):
    """
    검증 데이터에서 FAKE 기준 F1이 가장 높은 threshold를 찾는다.

    테스트 데이터에는 이 함수를 사용하지 않는다.

    반환:
        best_threshold
        best_f1
    """

    if len(scores) != len(labels):
        raise ValueError(
            "scores와 labels의 길이가 다릅니다."
        )

    if len(scores) == 0:
        raise ValueError(
            "Threshold를 정할 검증 데이터가 없습니다."
        )

    if len(set(labels)) < 2:
        raise ValueError(
            "검증 데이터에는 REAL과 FAKE가 모두 필요합니다."
        )

    # 점수 범위 안에서 후보 threshold 생성
    unique_scores = sorted(set(scores))

    # 서로 인접한 점수 사이의 중간값도 후보로 추가
    candidates = list(unique_scores)

    for left, right in zip(
        unique_scores[:-1],
        unique_scores[1:],
    ):
        candidates.append((left + right) / 2.0)

    # 모든 샘플을 REAL로 예측하는 경계도 후보에 추가
    candidates.append(max(unique_scores) + 1e-12)

    best_threshold = candidates[0]
    best_f1 = -1.0

    for threshold in candidates:

        predictions = predict_labels(
            scores,
            threshold,
        )

        metrics = calculate_metrics(
            labels,
            predictions,
        )

        f1 = metrics["f1_fake"]

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


# ============================================================
# 8. CSV 저장
# ============================================================
def save_results_csv(
    output_path,
    scores,
    labels,
    predictions,
):
    """
    샘플별 평가 결과를 CSV로 저장한다.

    저장 항목:
        label: 실제 라벨
        score: 이상 점수
        pred: 예측 라벨
        correct: 정답 여부
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "label",
            "score",
            "pred",
            "correct",
        ])

        for label, score, pred in zip(
            labels,
            scores,
            predictions,
        ):
            writer.writerow([
                label,
                score,
                pred,
                int(label == pred),
            ])

    print(f"CSV 저장 완료: {output_path.resolve()}")


# ============================================================
# 9. 평가 결과 출력
# ============================================================
def print_metrics(title, metrics):
    """평가 지표와 혼동 행렬을 출력한다."""

    print("\n" + "=" * 55)
    print(title)
    print("=" * 55)

    print(f"전체 샘플 수: {metrics['total']}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Precision:     {metrics['precision_fake']:.4f}")
    print(f"Recall:        {metrics['recall_fake']:.4f}")
    print(f"F1:            {metrics['f1_fake']:.4f}")

    print("\n혼동 행렬")
    print("                 예측 REAL   예측 FAKE")
    print(
        f"실제 REAL       {metrics['tn']:8d}"
        f"   {metrics['fp']:8d}"
    )
    print(
        f"실제 FAKE       {metrics['fn']:8d}"
        f"   {metrics['tp']:8d}"
    )


# ============================================================
# 10. 검증 + 테스트 평가
# ============================================================
def evaluate_model(
    model,
    val_loader,
    test_loader,
    output_dir="outputs/evaluation",
    device=None,
    start_t=200,
):
    """
    검증 데이터로 threshold를 선택하고,
    테스트 데이터에 해당 threshold를 적용한다.

    테스트 데이터로 threshold를 재조정하지 않는다.
    """

    if device is None:
        device = next(model.parameters()).device

    device = torch.device(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1) 검증 데이터 점수 계산
    # --------------------------------------------------------
    print("\n[1/3] 검증 데이터 평가")

    val_scores, val_labels = calculate_scores(
        model=model,
        data_loader=val_loader,
        device=device,
        start_t=start_t,
    )

    # --------------------------------------------------------
    # 2) 검증 데이터에서 threshold 결정
    # --------------------------------------------------------
    threshold, val_best_f1 = find_best_threshold(
        val_scores,
        val_labels,
    )

    print(f"\n선택된 threshold: {threshold:.8f}")
    print(f"검증 데이터 FAKE F1: {val_best_f1:.4f}")

    val_predictions = predict_labels(
        val_scores,
        threshold,
    )

    val_metrics = calculate_metrics(
        val_labels,
        val_predictions,
    )

    print_metrics(
        "검증 데이터 결과",
        val_metrics,
    )

    save_results_csv(
        output_dir / "validation_results.csv",
        val_scores,
        val_labels,
        val_predictions,
    )

    # --------------------------------------------------------
    # 3) 테스트 데이터 평가
    # --------------------------------------------------------
    print("\n[2/3] 테스트 데이터 평가")

    test_scores, test_labels = calculate_scores(
        model=model,
        data_loader=test_loader,
        device=device,
        start_t=start_t,
    )

    # 검증에서 정한 threshold를 그대로 사용
    test_predictions = predict_labels(
        test_scores,
        threshold,
    )

    test_metrics = calculate_metrics(
        test_labels,
        test_predictions,
    )

    print_metrics(
        "테스트 데이터 결과",
        test_metrics,
    )

    save_results_csv(
        output_dir / "test_results.csv",
        test_scores,
        test_labels,
        test_predictions,
    )

    # --------------------------------------------------------
    # 4) threshold 및 요약 결과 저장
    # --------------------------------------------------------
    summary_path = output_dir / "metrics.txt"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"threshold: {threshold:.10f}\n"
        )

        file.write(
            f"start_t: {start_t}\n"
        )

        file.write(
            f"validation_best_f1: {val_best_f1:.6f}\n"
        )

        file.write("\n[TEST]\n")

        for key, value in test_metrics.items():
            file.write(f"{key}: {value}\n")

    print(f"\n요약 저장 완료: {summary_path.resolve()}")

    return {
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }