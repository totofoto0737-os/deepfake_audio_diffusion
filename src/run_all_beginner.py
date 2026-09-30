"""
run_all.py
--------------------------------------------------
Diffusion 기반 딥보이스 탐지 전체 실행 스크립트

실행 흐름
1. protocol.py에서 REAL / FAKE 라벨 읽기
2. audio_root에서 음성 파일 검색
3. 파일 ID와 라벨 연결
4. REAL은 Train / Validation / Test로 분할
5. FAKE는 Validation / Test로 분할
6. REAL만 사용하여 Diffusion noise predictor 학습
7. 학습된 모델 checkpoint 저장
8. Validation에서 threshold 결정
9. 같은 threshold로 Test 평가
10. 결과 CSV / metrics 저장

현재 프로젝트의 라벨 규칙
    0 = REAL / bonafide
    1 = FAKE / spoof
"""

# argparse는 터미널에서 --epochs 같은 옵션을 받을 때 사용합니다.
import argparse
# json은 실행 결과를 사람이 읽기 쉬운 .json 파일로 저장할 때 사용합니다.
import json
# random은 데이터 순서를 섞을 때 사용합니다.
import random
# Path는 파일이나 폴더의 주소를 다루기 쉽게 해주는 Python 도구입니다.
from pathlib import Path

# PyTorch는 신경망을 만들고 학습시키는 데 사용하는 라이브러리입니다.
import torch
# DataLoader는 음성 데이터를 한 번에 여러 개(batch)씩 꺼내 줍니다.
from torch.utils.data import DataLoader

# 우리 프로젝트에서 공통으로 사용하는 숫자 설정을 가져옵니다.
from config import (
    BATCH_SIZE,
    HIDDEN_CHANNELS,
    IN_CHANNELS,
    LR,
    TIME_DIM,
    TIMESTEPS,
)
# AudioDataset은 음성 파일을 읽어서 Mel-spectrogram으로 바꿔 줍니다.
from dataset import AudioDataset
# evaluate_model은 학습이 끝난 모델을 validation/test 데이터로 평가합니다.
from evaluate import evaluate_model
# DiffusionModel은 실제로 노이즈를 배우는 신경망입니다.
from model import DiffusionModel
# protocol 파일에서 "이 파일은 REAL인가 FAKE인가"를 읽어옵니다.
from protocol import read_labels


# ============================================================
# 1. 랜덤 시드
# ============================================================

def set_seed(seed):
    # 같은 숫자를 사용하면 프로그램을 다시 실행해도
    # 데이터 섞기 등의 랜덤 동작을 최대한 똑같이 재현할 수 있습니다.
    """재현 가능한 데이터 분할과 학습을 위해 랜덤 시드를 설정한다."""

    # Python의 랜덤 숫자 생성기를 고정합니다.
    random.seed(seed)
    # PyTorch의 랜덤 숫자 생성기를 고정합니다.
    torch.manual_seed(seed)

    # CUDA를 사용할 수 있는지 확인합니다.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. 실행 장치
# ============================================================

def get_device():
    # 컴퓨터에 NVIDIA GPU가 있으면 GPU를 사용하고,
    # 없으면 일반 CPU를 사용합니다.
    """CUDA GPU가 있으면 GPU, 없으면 CPU를 사용한다."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# 3. 오디오 파일 검색
# ============================================================

def get_audio_files(root):
    # root 폴더 안을 살펴보면서 음성 파일을 모두 찾습니다.
    """
    root 아래의 음성 파일을 모두 찾는다.

    지원 확장자:
        .flac
        .wav
        .mp3
    """

    # 문자열로 받은 폴더 주소를 Path 객체로 바꿉니다.
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(
            f"오디오 폴더가 존재하지 않습니다: {root}"
        )

    # rglob("*")는 root 안의 하위 폴더까지 모두 살펴봅니다.
    # 그중 음성 파일 확장자를 가진 파일만 골라냅니다.
    audio_files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".flac", ".wav", ".mp3"}
    ]

    audio_files.sort()

    return audio_files


# ============================================================
# 4. 파일과 라벨 연결
# ============================================================

def build_labeled_items(audio_root, key_root):
    # 음성 파일 이름과 protocol에 적힌 정답을 연결합니다.
    # 예를 들어 "abc123.flac" -> "abc123" -> REAL(0) 같은 연결입니다.
    """
    오디오 파일의 stem(file_id)과 protocol label을 연결한다.

    반환:
        [(Path, label), ...]
    """

    # protocol 파일을 읽어 {파일ID: 라벨} 형태의 정보를 얻습니다.
    labels = read_labels(key_root)
    # 실제 음성 파일 목록을 가져옵니다.
    audio_files = get_audio_files(audio_root)

    labeled_items = []
    missing_label_count = 0

    for path in audio_files:
        # "abc123.flac"에서 확장자를 뺀 "abc123"만 가져옵니다.
        file_id = path.stem

        # protocol에 없는 파일은 정답을 알 수 없으므로 사용하지 않습니다.
        if file_id not in labels:
            missing_label_count += 1
            continue

        label = labels[file_id]

        if label not in (0, 1):
            raise ValueError(
                f"잘못된 라벨입니다: file_id={file_id}, label={label}"
            )

        labeled_items.append((path, label))

    if not labeled_items:
        raise RuntimeError(
            "오디오 파일과 protocol label이 하나도 연결되지 않았습니다.\n"
            f"audio_root={audio_root}\n"
            f"key_root={key_root}\n"
            "파일 ID(stem)와 protocol의 file_id 형식을 확인하세요."
        )

    print(f"전체 오디오 파일: {len(audio_files)}")
    print(f"라벨 연결 성공:   {len(labeled_items)}")
    print(f"라벨 없음:         {missing_label_count}")

    return labeled_items


# ============================================================
# 5. REAL / FAKE 분리
# ============================================================

def split_by_label(items):
    # 전체 데이터를 REAL과 FAKE 두 상자로 나눕니다.
    """REAL(0)과 FAKE(1)를 분리한다."""

    real_items = [item for item in items if item[1] == 0]
    fake_items = [item for item in items if item[1] == 1]

    return real_items, fake_items


# ============================================================
# 6. Train / Validation / Test 분할
# ============================================================

def make_splits(

    real_items,
    fake_items,
    max_train_real,
    max_val_real,
    max_val_fake,
    max_test_real,
    max_test_fake,
):
    """
    현재 프로젝트의 anomaly-detection 구조에 맞게 분할한다.

    Train:
        REAL만 사용

    Validation:
        REAL + FAKE

    Test:
        REAL + FAKE
    """

    if max_train_real <= 0:
        raise ValueError("--max-train-real은 1 이상이어야 합니다.")

    if max_val_real <= 0:
        raise ValueError("--max-val-real은 1 이상이어야 합니다.")

    if max_val_fake <= 0:
        raise ValueError("--max-val-fake는 1 이상이어야 합니다.")

    if max_test_real <= 0:
        raise ValueError("--max-test-real은 1 이상이어야 합니다.")

    if max_test_fake <= 0:
        raise ValueError("--max-test-fake는 1 이상이어야 합니다.")

    required_real = (
        max_train_real
        + max_val_real
        + max_test_real
    )

    required_fake = (
        max_val_fake
        + max_test_fake
    )

    if len(real_items) < required_real:
        raise ValueError(
            "REAL 데이터가 부족합니다.\n"
            f"필요: {required_real}\n"
            f"현재: {len(real_items)}\n"
            "max_train_real / max_val_real / max_test_real 값을 줄이세요."
        )

    if len(fake_items) < required_fake:
        raise ValueError(
            "FAKE 데이터가 부족합니다.\n"
            f"필요: {required_fake}\n"
            f"현재: {len(fake_items)}\n"
            "max_val_fake / max_test_fake 값을 줄이세요."
        )

    # 먼저 학습용 REAL 음성을 가져옵니다.
    train_real = real_items[
        :max_train_real
    ]

    # validation에서는 REAL도 일부 사용합니다.
    val_real = real_items[
        max_train_real:
        max_train_real + max_val_real
    ]

    # test에서도 REAL을 사용합니다.
    test_real = real_items[
        max_train_real + max_val_real:
        max_train_real + max_val_real + max_test_real
    ]

    # validation에서 사용할 FAKE 음성입니다.
    val_fake = fake_items[
        :max_val_fake
    ]

    # test에서 사용할 FAKE 음성입니다.
    test_fake = fake_items[
        max_val_fake:
        max_val_fake + max_test_fake
    ]

    val_data = val_real + val_fake
    test_data = test_real + test_fake

    return (
        train_real,
        val_data,
        test_data,
    )


# ============================================================
# 7. DataLoader
# ============================================================

def make_dataloaders(
    train_data,
    val_data,
    test_data,
    batch_size,
    num_workers,
):
    """\n    Train / Validation / Test DataLoader를 만듭니다.\n\n    DataLoader는 많은 음성을 한 번에 하나씩 처리하지 않고\n    batch_size개씩 묶어서 모델에게 전달해 줍니다.\n    """

    if batch_size <= 0:
        raise ValueError("--batch-size는 1 이상이어야 합니다.")

    # 학습 데이터는 순서를 섞습니다.
    # 그래야 모델이 항상 같은 순서로 음성을 보지 않습니다.
    train_loader = DataLoader(
        AudioDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # validation은 평가용이므로 순서를 섞지 않습니다.
    val_loader = DataLoader(
        AudioDataset(val_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # test도 평가용이므로 순서를 섞지 않습니다.
    test_loader = DataLoader(
        AudioDataset(test_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 8. 한 epoch 학습
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    total_epochs,
    grad_clip=1.0,
):
    """
    REAL Mel-spectrogram만 사용하여 diffusion noise predictor를 학습한다.
    """

    # 학습 모드로 바꿉니다.
    model.train()

    total_loss = 0.0
    total_samples = 0

    for step, (x, _label, _path) in enumerate(
        train_loader,
        start=1,
    ):
        # 음성에서 만든 Mel-spectrogram을 GPU 또는 CPU로 옮깁니다.
        x = x.to(
            device,
            non_blocking=True,
        )

        # Diffusion에서는 음성에 노이즈를 조금 넣을 수도 있고
        # 많이 넣을 수도 있습니다.
        # 그래서 0부터 TIMESTEPS-1 사이의 숫자를 무작위로 뽑습니다.
        # 이 숫자를 timestep(t)이라고 부릅니다.
        #
        # 각 샘플마다 random timestep 선택
        t = torch.randint(
            low=0,
            high=TIMESTEPS,
            size=(x.size(0),),
            device=device,
            dtype=torch.long,
        )

        # 원래 깨끗한 데이터 x_0에 노이즈를 섞어서 x_t를 만듭니다.
        # 그리고 실제로 넣은 노이즈도 함께 받아옵니다.
        #
        # x_0 -> x_t
        noisy_x, noise = model.add_noise(
            x,
            t,
        )

        # 모델에게 "이 안에 들어 있는 노이즈가 무엇인지 맞혀봐"라고 시킵니다.
        #
        # x_t에 포함된 noise 예측
        predicted_noise = model.predict_noise(
            noisy_x,
            t,
        )

        # 정답 노이즈와 모델의 예측 노이즈가 얼마나 다른지 계산합니다.
        # 이 숫자가 작을수록 모델이 노이즈를 잘 맞힌 것입니다.
        #
        # DDPM noise prediction loss
        loss = (
            predicted_noise - noise
        ).pow(2).mean()

        # 지난 문제의 기억(gradient)을 지웁니다.
        optimizer.zero_grad(set_to_none=True)

        # "어느 부분을 얼마나 고쳐야 하는지"를 계산합니다.
        loss.backward()

        # 한 번에 너무 크게 모델이 바뀌지 않도록 제한합니다.
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        # 방금 계산한 방향대로 모델의 숫자(가중치)를 실제로 수정합니다.
        optimizer.step()

        batch_size = x.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if step % 20 == 0 or step == len(train_loader):
            print(
                f"  Epoch [{epoch}/{total_epochs}] "
                f"Step [{step}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f}"
            )

    return total_loss / max(total_samples, 1)


# ============================================================
# 9. 전체 학습
# ============================================================

def train_model(
    model,
    train_loader,
    optimizer,
    device,
    epochs,
    grad_clip=1.0,
):
    """\n    여러 epoch 동안 반복해서 모델을 공부시킵니다.\n\n    epoch가 10이면 전체 학습 데이터를 10번 둘러봅니다.\n    """

    if epochs <= 0:
        raise ValueError("--epochs는 1 이상이어야 합니다.")

    history = []

    print("\n" + "=" * 70)
    print("학습 시작")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        average_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            grad_clip=grad_clip,
        )

        history.append(
            {
                "epoch": epoch,
                "loss": average_loss,
            }
        )

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Average Loss: {average_loss:.6f}"
        )

    return history


# ============================================================
# 10. Checkpoint 저장
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    checkpoint_path,
    epoch,
    history,
):
    """
    모델 state뿐 아니라 학습 설정과 optimizer state도 함께 저장한다.

    evaluate.py의 load_model()은 checkpoint 안에
    model_state_dict가 있는 형식도 지원한다.
    """

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "config": {
            "in_channels": IN_CHANNELS,
            "hidden_channels": HIDDEN_CHANNELS,
            "time_dim": TIME_DIM,
            "timesteps": TIMESTEPS,
            "learning_rate": LR,
        },
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print(
        f"\n체크포인트 저장 완료: "
        f"{checkpoint_path.resolve()}"
    )


# ============================================================
# 11. 학습 기록 저장
# ============================================================

def save_training_history(history, output_dir):
    # epoch마다 loss가 어떻게 변했는지 JSON 파일로 저장합니다.
    """학습 loss를 JSON으로 저장한다."""

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = output_dir / "training_history.json"

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"학습 기록 저장 완료: "
        f"{history_path.resolve()}"
    )


# ============================================================
# 12. 데이터 분할 정보 저장
# ============================================================

def save_split_summary(
    output_dir,
    train_data,
    val_data,
    test_data,
):
    """\n    Train / Validation / Test에 음성이 몇 개씩 들어갔는지 저장합니다.\n    나중에 "내 데이터가 제대로 나뉘었나?" 확인할 때 매우 유용합니다.\n    """

    def count_labels(items):
        real_count = sum(
            1 for _, label in items if label == 0
        )
        fake_count = sum(
            1 for _, label in items if label == 1
        )
        return real_count, fake_count

    train_real, train_fake = count_labels(train_data)
    val_real, val_fake = count_labels(val_data)
    test_real, test_fake = count_labels(test_data)

    summary = {
        "train": {
            "total": len(train_data),
            "real": train_real,
            "fake": train_fake,
        },
        "validation": {
            "total": len(val_data),
            "real": val_real,
            "fake": val_fake,
        },
        "test": {
            "total": len(test_data),
            "real": test_real,
            "fake": test_fake,
        },
    }

    path = Path(output_dir) / "split_summary.json"

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"분할 정보 저장 완료: "
        f"{path.resolve()}"
    )


# ============================================================
# 13. main
# ============================================================

def main():
    # 프로그램을 실제로 실행하는 중심 함수입니다.
    # 아래에서 데이터 준비 -> 학습 -> 평가를 순서대로 합니다.
    # 터미널에서 값을 쉽게 바꿀 수 있도록 명령어 옵션을 만듭니다.
    parser = argparse.ArgumentParser(
        description=(
            "Diffusion 기반 딥보이스 탐지 "
            "학습 + validation + test 실행"
        )
    )

    # 데이터
    parser.add_argument(
        "--audio-root",
        required=True,
        help="오디오 파일이 들어 있는 최상위 폴더",
    )

    parser.add_argument(
        "--key-root",
        required=True,
        help="protocol key 파일이 들어 있는 폴더",
    )

    # 출력
    parser.add_argument(
        "--out-dir",
        default="outputs/diffusion_run",
        help="결과 저장 폴더",
    )

    # 학습
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="학습 epoch 수",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="batch size",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help="AdamW learning rate",
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="gradient clipping 최대 norm",
    )

    # 데이터 개수
    parser.add_argument(
        "--max-train-real",
        type=int,
        default=1000,
        help="학습에 사용할 REAL 최대 개수",
    )

    parser.add_argument(
        "--max-val-real",
        type=int,
        default=200,
        help="validation REAL 최대 개수",
    )

    parser.add_argument(
        "--max-val-fake",
        type=int,
        default=200,
        help="validation FAKE 최대 개수",
    )

    parser.add_argument(
        "--max-test-real",
        type=int,
        default=200,
        help="test REAL 최대 개수",
    )

    parser.add_argument(
        "--max-test-fake",
        type=int,
        default=200,
        help="test FAKE 최대 개수",
    )

    # 평가
    # 평가에서 사용할 노이즈의 세기를 정합니다.
    # TIMESTEPS가 100이면 사용할 수 있는 숫자는 0~99입니다.
    # 예전 코드의 200은 범위를 벗어나므로 사용하지 않습니다.
    parser.add_argument(
        "--start-t",
        type=int,
        default=50,
        help=(
            "평가 시 reconstruction을 시작할 diffusion timestep. "
            f"0 <= start_t < TIMESTEPS({TIMESTEPS})"
        ),
    )

    # 기타
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker 수",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # 기본 인자 검증
    # --------------------------------------------------------

    if not 0 <= args.start_t < TIMESTEPS:
        raise ValueError(
            f"--start-t는 0부터 {TIMESTEPS - 1} 사이여야 합니다. "
            f"현재 값: {args.start_t}"
        )

    if args.num_workers < 0:
        raise ValueError(
            "--num-workers는 0 이상이어야 합니다."
        )

    if args.lr <= 0:
        raise ValueError(
            "--lr은 0보다 커야 합니다."
        )

    if args.grad_clip <= 0:
        raise ValueError(
            "--grad-clip은 0보다 커야 합니다."
        )

    # --------------------------------------------------------
    # 출력 폴더
    # --------------------------------------------------------

    output_dir = Path(args.out_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Seed / device
    # --------------------------------------------------------

    set_seed(args.seed)
    device = get_device()

    print("=" * 70)
    print("Diffusion Deepfake / Deepvoice Detection")
    print("=" * 70)
    print(f"Device:     {device}")
    print(f"TIMESTEPS:  {TIMESTEPS}")
    print(f"Start t:    {args.start_t}")
    print(f"Seed:       {args.seed}")
    print(f"Output:     {output_dir.resolve()}")

    # --------------------------------------------------------
    # 데이터 읽기
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("데이터 준비")
    print("=" * 70)

    all_data = build_labeled_items(
        audio_root=args.audio_root,
        key_root=args.key_root,
    )

    # 같은 종류의 음성이 한쪽에 몰리지 않도록 순서를 섞습니다.
    random.shuffle(all_data)

    real_data, fake_data = split_by_label(
        all_data
    )

    print(f"\nREAL 전체: {len(real_data)}")
    print(f"FAKE 전체: {len(fake_data)}")

    train_data, val_data, test_data = make_splits(
        real_items=real_data,
        fake_items=fake_data,
        max_train_real=args.max_train_real,
        max_val_real=args.max_val_real,
        max_val_fake=args.max_val_fake,
        max_test_real=args.max_test_real,
        max_test_fake=args.max_test_fake,
    )

    print("\n최종 데이터 분할")
    print(f"Train      : {len(train_data)} (REAL only)")
    print(f"Validation : {len(val_data)}")
    print(f"Test       : {len(test_data)}")

    save_split_summary(
        output_dir=output_dir,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader, val_loader, test_loader = make_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # --------------------------------------------------------
    # 모델
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("모델 생성")
    print("=" * 70)

    model = DiffusionModel(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_CHANNELS,
        time_dim=TIME_DIM,
        timesteps=TIMESTEPS,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    print(
        f"IN_CHANNELS={IN_CHANNELS}, "
        f"HIDDEN_CHANNELS={HIDDEN_CHANNELS}, "
        f"TIME_DIM={TIME_DIM}"
    )
    print(f"Learning rate: {args.lr}")

    # --------------------------------------------------------
    # 학습
    # --------------------------------------------------------

    history = train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
    )

    save_training_history(
        history=history,
        output_dir=output_dir,
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint_path = output_dir / "model.pt"

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
        epoch=args.epochs,
        history=history,
    )

    # --------------------------------------------------------
    # 평가
    # --------------------------------------------------------

    evaluation_dir = output_dir / "evaluation"

    print("\n" + "=" * 70)
    print("Validation / Test 평가")
    print("=" * 70)

    # 중요:
    # config.py의 TIMESTEPS=100이므로 start_t=200을 사용하면 안 된다.
    # validation에서 threshold를 정하고 test에는 같은 threshold를 적용한다.
    results = evaluate_model(
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=evaluation_dir,
        device=device,
        start_t=args.start_t,
    )

    # --------------------------------------------------------
    # 최종 요약
    # --------------------------------------------------------

    summary = {
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "timesteps": TIMESTEPS,
        "start_t": args.start_t,
        "checkpoint": str(checkpoint_path.resolve()),
        "evaluation_dir": str(evaluation_dir.resolve()),
        "threshold": results["threshold"],
        "validation_metrics": results["validation_metrics"],
        "test_metrics": results["test_metrics"],
    }

    summary_path = output_dir / "run_summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("전체 실행 완료")
    print("=" * 70)
    print(f"Model      : {checkpoint_path.resolve()}")
    print(f"Evaluation : {evaluation_dir.resolve()}")
    print(f"Threshold  : {results['threshold']:.8f}")
    print(f"Summary    : {summary_path.resolve()}")


if __name__ == "__main__":
    main()
