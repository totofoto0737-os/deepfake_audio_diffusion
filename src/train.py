"""
train.py
--------------------------------------------------
Diffusion 기반 딥보이스 탐지 모델의 학습 코드

학습 방식:
    REAL 음성만 사용하여 노이즈 예측 모델을 학습한다.

학습 순서:
    1. REAL Mel-spectrogram 불러오기
    2. 무작위 Diffusion 시간 단계 t 선택
    3. 원본에 노이즈 추가
    4. 모델이 노이즈 예측
    5. 실제 노이즈와 예측 노이즈의 MSE 계산
    6. 역전파 및 가중치 업데이트
    7. 모델 체크포인트 저장

주의:
    이 파일은 데이터셋을 직접 생성하지 않는다.
    외부에서 준비한 train_loader를 전달받는다.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DiffusionModel


# ============================================================
# 1. 학습에 사용할 장치 설정
# ============================================================
def get_device():
    """
    CUDA GPU를 사용할 수 있으면 GPU를 사용하고,
    그렇지 않으면 CPU를 사용한다.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# 2. DataLoader에서 Mel-spectrogram 꺼내기
# ============================================================
def extract_mel(batch):
    """
    DataLoader가 반환하는 batch에서 Mel-spectrogram을 꺼낸다.

    지원하는 형태:
        Tensor
        (mel, label)
        [mel, label]
        {"mel": mel}
        {"x": mel}
        {"features": mel}

    라벨은 학습에 사용하지 않는다.
    """

    if isinstance(batch, torch.Tensor):
        mel = batch

    elif isinstance(batch, (tuple, list)):
        # 일반적인 Dataset은 (데이터, 라벨)을 반환한다.
        mel = batch[0]

    elif isinstance(batch, dict):
        if "mel" in batch:
            mel = batch["mel"]
        elif "x" in batch:
            mel = batch["x"]
        elif "features" in batch:
            mel = batch["features"]
        else:
            raise KeyError(
                "batch 딕셔너리에서 Mel 데이터를 찾을 수 없습니다. "
                "'mel', 'x', 'features' 중 하나가 필요합니다."
            )

    else:
        raise TypeError(
            f"지원하지 않는 batch 형식입니다: {type(batch)}"
        )

    # 모델은 [B, 1, mel_bins, time_frames] 형태를 기대한다.
    #
    # [B, mel_bins, time_frames]라면 채널 차원을 추가한다.
    if mel.ndim == 3:
        mel = mel.unsqueeze(1)

    if mel.ndim != 4:
        raise ValueError(
            "Mel 데이터는 3차원 또는 4차원이어야 합니다. "
            f"현재 shape: {tuple(mel.shape)}"
        )

    return mel.float()


# ============================================================
# 3. 한 번의 학습 단계
# ============================================================
def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    epoch,
    total_epochs,
):
    """
    전체 학습 데이터를 한 번 순회한다.

    한 epoch에서 수행하는 작업:
        REAL Mel -> 노이즈 추가 -> 노이즈 예측
        -> 손실 계산 -> 역전파 -> 가중치 업데이트
    """

    # 학습 모드로 전환
    model.train()

    total_loss = 0.0
    total_samples = 0

    for step, batch in enumerate(train_loader, start=1):

        # ----------------------------------------------------
        # 1) Mel-spectrogram 가져오기
        # ----------------------------------------------------
        x_0 = extract_mel(batch).to(device)

        # ----------------------------------------------------
        # 2) 무작위 Diffusion 시간 단계 선택
        # ----------------------------------------------------
        # 각 음성마다 서로 다른 시간 단계를 선택할 수 있다.
        #
        # 예:
        #   첫 번째 음성: t=120
        #   두 번째 음성: t=750
        #   세 번째 음성: t=430
        #
        # 시간 단계에 따라 추가되는 노이즈의 양이 달라진다.
        t = torch.randint(
            low=0,
            high=model.timesteps,
            size=(x_0.shape[0],),
            device=device,
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # 3) 원본에 노이즈 추가
        # ----------------------------------------------------
        # x_0: 원본 Mel-spectrogram
        # x_t: 노이즈가 추가된 Mel-spectrogram
        # noise: 실제로 추가한 노이즈
        x_t, noise = model.add_noise(x_0, t)

        # ----------------------------------------------------
        # 4) 모델이 노이즈 예측
        # ----------------------------------------------------
        predicted_noise = model.predict_noise(x_t, t)

        # ----------------------------------------------------
        # 5) 손실 계산
        # ----------------------------------------------------
        # 실제 노이즈와 예측 노이즈의 차이를 계산한다.
        #
        # 손실이 작을수록 모델이 노이즈를 잘 예측한다.
        loss = nn.functional.mse_loss(
            predicted_noise,
            noise,
        )

        # ----------------------------------------------------
        # 6) 역전파 및 가중치 업데이트
        # ----------------------------------------------------
        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        # 큰 gradient로 학습이 불안정해지는 것을 완화
        torch.nn.utils.clip_grad_norm