# ============================================================
# 필요한 라이브러리 가져오기
# ============================================================

import argparse
import random
import json

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from protocol import read_labels
from dataset import AudioDataset
from model import TinyUNet
from config import *


# ============================================================
# 1. 오디오 파일 찾기
# ============================================================

def get_audio_files(root):

    """
    지정한 폴더 안에서
    .flac / .wav / .mp3 파일을 모두 찾는다.
    """

    root = Path(root)

    audio_files = []

    # root 폴더 안의 모든 파일을 확인
    for path in root.rglob("*"):

        # 파일 확장자 확인
        extension = path.suffix.lower()

        # 음성 파일이면 저장
        if extension in [".flac", ".wav", ".mp3"]:
            audio_files.append(path)

    # 파일 이름 순서대로 정렬
    audio_files.sort()

    return audio_files


# ============================================================
# 2. Diffusion 모델의 이상 점수 계산
# ============================================================

def calculate_score(model, x):

    """
    입력된 Mel-spectrogram이
    모델이 학습한 REAL 음성과 얼마나 다른지 계산한다.

    점수가 높을수록 → 이상할 가능성이 높음
    점수가 낮을수록 → REAL과 비슷할 가능성이 높음
    """

    # --------------------------------------------------------
    # 1. 사용할 diffusion timestep 결정
    # --------------------------------------------------------

    # 모든 데이터에 같은 timestep을 사용한다.
    #
    # 예:
    # TIMESTEPS = 1000이면
    # 500번째 timestep 사용
    t = torch.full(
        (x.size(0),),
        TIMESTEPS // 2,
        device=x.device,
        dtype=torch.long
    )


    # --------------------------------------------------------
    # 2. Beta 값 만들기
    # --------------------------------------------------------

    # diffusion 과정에서
    # 얼마나 noise를 추가할지 결정하는 값
    beta = torch.linspace(
        1e-4,
        0.02,
        TIMESTEPS,
        device=x.device
    )


    # --------------------------------------------------------
    # 3. alpha 누적값 계산
    # --------------------------------------------------------

    alpha = 1 - beta

    alpha_bar = torch.cumprod(
        alpha,
        dim=0
    )


    # 현재 timestep에 해당하는 값만 가져온다.
    alpha_bar_t = alpha_bar[t]

    # 계산하기 편하게 차원 변경
    alpha_bar_t = alpha_bar_t.view(
        -1, 1, 1, 1
    )


    # --------------------------------------------------------
    # 4. 랜덤 noise 만들기
    # --------------------------------------------------------

    noise = torch.randn_like(x)


    # --------------------------------------------------------
    # 5. 원래 Mel-spectrogram에 noise 추가
    # --------------------------------------------------------

    noisy_x = (
        torch.sqrt(alpha_bar_t) * x
        +
        torch.sqrt(1 - alpha_bar_t) * noise
    )


    # --------------------------------------------------------
    # 6. 모델에게 noise를 예측하게 함
    # --------------------------------------------------------

    predicted_noise = model(
        noisy_x,
        t.float() / TIMESTEPS
    )


    # --------------------------------------------------------
    # 7. 예측한 noise를 이용해서
    #    원래 Mel-spectrogram을 복원
    # --------------------------------------------------------

    reconstructed_x = (
        noisy_x
        -
        torch.sqrt(1 - alpha_bar_t) * predicted_noise
    ) / torch.sqrt(alpha_bar_t)


    # --------------------------------------------------------
    # 8. 원본과 복원 결과의 차이 계산
    # --------------------------------------------------------

    # 값이 클수록
    # 원래 데이터와 복원한 데이터가 많이 다르다는 뜻
    score = (
        x
        -
        reconstructed_x.clamp(0, 1)
    ).abs().mean(
        dim=(1, 2, 3)
    )


    return score


# ============================================================
# 3. 프로그램 시작 -> 명령어로 실행시킬 때 시작
# ============================================================

def main():

    # ========================================================
    # 실행할 때 입력받을 옵션 설정-> 명령어로 넣어준 옵션대로 실행하도록 만듬
    # ========================================================

    parser = argparse.ArgumentParser()


    # 오디오 데이터 폴더를 "--audio-root" 명령어로 받는다.
    parser.add_argument(
        "--audio-root",
        required=True
    )


    # 라벨 파일이 있는 폴더의 정보를 "--key-root" 명령어로 받는다. 
    parser.add_argument(
        "--key-root",
        required=True
    )


    # 결과를 저장할 폴더경로를 "--out-dir" 명령어로 받는다.
    parser.add_argument(
        "--out-dir",
        default="outputs/diffusion_run"
    )


    # 학습 횟수를 "--epochs" 명령어로 받는다. 
    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )


    # 한 번에 학습할 데이터 개수를 "--batch-size" 명령어로 받는다. 
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE
    )


    # 학습에 사용할 REAL 데이터 수를 "--max-train-real" 명령어로 받는다. 
    parser.add_argument(
        "--max-train-real",
        type=int,
        default=1000
    )


    # Validation REAL 데이터 수를 "--max-val-real" 명령어로 받는다. 
    parser.add_argument(
        "--max-val-real",
        type=int,
        default=200
    )


    # Validation FAKE 데이터 수를 "--max-val-real" 명령어로 받는다. 
    parser.add_argument(
        "--max-val-fake",
        type=int,
        default=200
    )


    # Test REAL 데이터 수를 "--max-test-real" 명령어로 받는다. 
    parser.add_argument(
        "--max-test-real",
        type=int,
        default=200
    )


    # Test FAKE 데이터 수를 "--max-test-fake" 명령어로 받는다. 
    parser.add_argument(
        "--max-test-fake",
        type=int,
        default=200
    )


    # 실제 명령어에서 입력한 값을 저장한다. : '--epochs' 10 -> args.epochs = 10
    args = parser.parse_args()


    # ========================================================
    # 4. 랜덤 고정
    # ========================================================

    # 실행할 때마다 비슷한 데이터를 뽑도록
    # 랜덤값을 고정한다.
    random.seed(42)


    # ========================================================
    # 5. 라벨 읽기
    # ========================================================

    labels = read_labels(
        args.key_root
    )


    # ========================================================
    # 6. 오디오 파일 찾기
    # ========================================================

    audio_files = get_audio_files(
        args.audio_root
    )


    # ========================================================
    # 7. 오디오와 라벨 연결
    # ========================================================

    all_data = []

    for path in audio_files:

        # 파일 이름에서 확장자를 제거
        # ex) LA_E_9332881.flac -> LA_E_9332881
        file_id = path.stem


        # 라벨 파일에 해당 파일이 있는지 확인
        if file_id in labels:

            label = labels[file_id]

            # (파일 경로, 라벨) 형태로 저장
            all_data.append(
                (path, label)
            )


    # ========================================================
    # 8. 데이터 순서를 섞음
    # ========================================================

    random.shuffle(
        all_data
    )


    # ========================================================
    # 9. REAL / FAKE 분리
    # ========================================================

    real_data = []
    fake_data = []


    for item in all_data:

        path, label = item

        if label == 0:

            # REAL
            real_data.append(item)

        elif label == 1:

            # FAKE
            fake_data.append(item)


    # ========================================================
    # 10. Train / Validation / Test 데이터 만들기
    # ========================================================

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    #
    train_data = real_data[
        :args.max_train_real
    ]


    # --------------------------------------------------------
    # Validation REAL
    # --------------------------------------------------------

    val_real = real_data[
        args.max_train_real:
        args.max_train_real + args.max_val_real
    ]


    # --------------------------------------------------------
    # Validation FAKE
    # --------------------------------------------------------

    val_fake = fake_data[
        :args.max_val_fake
    ]


    # --------------------------------------------------------
    # Test REAL
    # --------------------------------------------------------

    test_real = real_data[
        args.max_train_real + args.max_val_real:
        args.max_train_real
        + args.max_val_real
        + args.max_test_real
    ]


    # --------------------------------------------------------
    # Test FAKE
    # --------------------------------------------------------

    test_fake = fake_data[
        args.max_val_fake:
        args.max_val_fake
        + args.max_test_fake
    ]


    # Test에는 REAL + FAKE 둘 다 들어간다.
    test_data = test_real + test_fake


    # ========================================================
    # 11. 사용할 장치 결정
    # ========================================================

    if torch.cuda.is_available():

        device = torch.device("cuda")

    else:

        device = torch.device("cpu")


    print(
        "Device:",
        device
    )

    print(
        "Train REAL:",
        len(train_data)
    )

    print(
        "Validation:",
        len(val_real) + len(val_fake)
    )

    print(
        "Test:",
        len(test_data)
    )


    # ========================================================
    # 12. DataLoader 만들기
    # ========================================================

    train_loader = DataLoader(
        AudioDataset(train_data),

        batch_size=args.batch_size,

        # 학습할 때 데이터 순서를 섞음
        shuffle=True,

        # Windows에서 안전하게 사용
        num_workers=0
    )


    # ========================================================
    # 13. Diffusion 모델 만들기
    # ========================================================

    model = TinyUNet()

    # CPU 또는 GPU로 이동
    model = model.to(device)


    # ========================================================
    # 14. Optimizer 만들기
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )


    # ========================================================
    # 15. 실제 학습 시작
    # ========================================================

    for epoch in range(args.epochs):

        # 학습 모드
        model.train()

        total_loss = 0


        # ----------------------------------------------------
        # Train 데이터 하나씩 가져오기
        # ----------------------------------------------------

        for x, label, path in train_loader:

            # Mel-spectrogram을 CPU/GPU로 이동
            x = x.to(device)


            # ------------------------------------------------
            # 랜덤 timestep 선택
            # ------------------------------------------------

            t = torch.randint(
                0,
                TIMESTEPS,
                (x.size(0),),
                device=device
            )


            # ------------------------------------------------
            # Beta 생성
            # ------------------------------------------------

            beta = torch.linspace(
                1e-4,
                0.02,
                TIMESTEPS,
                device=device
            )


            # ------------------------------------------------
            # Alpha 계산
            # ------------------------------------------------

            alpha = 1 - beta


            # ------------------------------------------------
            # Alpha 누적 계산
            # ------------------------------------------------

            alpha_bar = torch.cumprod(
                alpha,
                dim=0
            )


            # 현재 timestep의 alpha_bar
            alpha_bar_t = alpha_bar[t]


            # 차원 맞추기
            alpha_bar_t = alpha_bar_t.view(
                -1, 1, 1, 1
            )


            # ------------------------------------------------
            # 랜덤 noise 생성
            # ------------------------------------------------

            noise = torch.randn_like(x)


            # ------------------------------------------------
            # Mel-spectrogram에 noise 추가
            # ------------------------------------------------

            noisy_x = (
                torch.sqrt(alpha_bar_t) * x
                +
                torch.sqrt(1 - alpha_bar_t) * noise
            )


            # ------------------------------------------------
            # 모델이 noise를 예측
            # ------------------------------------------------

            predicted_noise = model(
                noisy_x,
                t.float() / TIMESTEPS
            )


            # ------------------------------------------------
            # 실제 noise와 예측 noise 비교
            # ------------------------------------------------

            loss = (
                predicted_noise - noise
            ).pow(2).mean()


            # ------------------------------------------------
            # 이전 gradient 초기화
            # ------------------------------------------------

            optimizer.zero_grad()


            # ------------------------------------------------
            # 오차를 이용해서 역전파
            # ------------------------------------------------

            loss.backward()


            # ------------------------------------------------
            # 너무 큰 gradient 방지
            # ------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1
            )


            # ------------------------------------------------
            # 모델의 가중치 업데이트
            # ------------------------------------------------

            optimizer.step()


            # loss 누적
            total_loss += loss.item()


        # ----------------------------------------------------
        # Epoch 결과 출력
        # ----------------------------------------------------

        average_loss = (
            total_loss /
            max(1, len(train_loader))
        )


        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={average_loss:.6f}"
        )


    # ========================================================
    # 16. Validation/Test 점수 계산 함수
    # ========================================================

    def collect_scores(items):

        # Dataset 만들기
        dataset = AudioDataset(items)


        # DataLoader 만들기
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )


        rows = []


        # 평가 모드
        model.eval()


        # 평가할 때 gradient 계산하지 않음
        with torch.no_grad():

            for x, label, path in loader:

                # score 계산
                scores = calculate_score(
                    model,
                    x.to(device)
                )


                # Tensor → NumPy
                scores = scores.cpu().numpy()


                # 결과 저장
                for file_path, file_label, score_value in zip(
                    path,
                    label,
                    scores
                ):

                    rows.append(
                        {
                            "path": file_path,
                            "label": int(file_label),
                            "score": float(score_value)
                        }
                    )


        # 표 형태로 변환
        return pd.DataFrame(rows)


    # ========================================================
    # 17. Validation 점수 계산
    # ========================================================

    validation_data = (
        val_real +
        val_fake
    )


    validation_result = collect_scores(
        validation_data
    )


    # ========================================================
    # 18. Threshold 계산
    # ========================================================

    # Validation의 REAL 데이터만 가져온다.
    real_scores = validation_result[
        validation_result.label == 0
    ].score


    # REAL 점수 중
    # REAL_PERCENTILE 위치의 값을 threshold로 사용
    threshold = float(
        np.percentile(
            real_scores,
            REAL_PERCENTILE
        )
    )


    # ========================================================
    # 19. Test 점수 계산
    # ========================================================

    test_result = collect_scores(
        test_data
    )


    # ========================================================
    # 20. Threshold로 REAL / FAKE 판단
    # ========================================================

    # score가 threshold보다 크면
    # → FAKE = 1
    #
    # score가 threshold보다 작거나 같으면
    # → REAL = 0

    test_result["pred"] = (
        test_result["score"] > threshold
    ).astype(int)


    # ========================================================
    # 21. 정답인지 확인
    # ========================================================

    test_result["correct"] = (
        test_result["pred"]
        ==
        test_result["label"]
    )


    # ========================================================
    # 22. 결과 저장 폴더 만들기
    # ========================================================

    output_dir = Path(
        args.out_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # 23. 학습된 모델 저장
    # ========================================================

    torch.save(
        model.state_dict(),
        output_dir / "model.pt"
    )


    # ========================================================
    # 24. Test 결과 CSV 저장
    # ========================================================

    test_result.to_csv(
        output_dir / "scores.csv",
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # 25. Threshold 정보 저장
    # ========================================================

    threshold_info = {
        "threshold": threshold,
        "percentile": REAL_PERCENTILE
    }


    with open(
        output_dir / "threshold.json",
        "w"
    ) as file:

        json.dump(
            threshold_info,
            file,
            indent=2
        )


    # ========================================================
    # 26. 결과 출력
    # ========================================================

    print(
        "Threshold:",
        threshold
    )

    print(
        "Saved:",
        output_dir
    )


# ============================================================
# 이 파일을 직접 실행했을 때만 main() 실행 ->다른 파일에서 run_all파일 함수나 변수를 import해도 또 실행하지 않도록 조치
# ============================================================

if __name__ == "__main__":

    main()