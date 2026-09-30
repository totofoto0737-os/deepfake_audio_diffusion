"""
model.py
--------------------------------------------------
Diffusion 기반 딥보이스 탐지 모델

주요 기능
1. 시간 t를 임베딩하는 TimeEmbedding
2. 현재 Mel-spectrogram의 노이즈를 예측하는 NoisePredictor
3. 정방향 확산: 원본 데이터에 노이즈 추가
4. 역방향 확산: 노이즈를 조금씩 제거하여 데이터 복원

입력 데이터 형태:
    [batch_size, 1, mel_bins, time_frames]

예:
    [16, 1, 80, 128]

주의:
    이 파일은 모델과 Diffusion 수식을 구현한다.
    실제 데이터 로딩과 학습 반복문은 train.py에서 담당한다.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 시간 임베딩
# ============================================================
class TimeEmbedding(nn.Module):
    """
    Diffusion의 시간 단계 t를 신경망이 이해할 수 있는 벡터로 변환한다.

    Diffusion에서는 t=0, t=1, ...에 따라 노이즈의 양이 다르다.
    따라서 모델에게 현재 몇 번째 단계인지 알려줘야 한다.
    """

    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        # 시간 임베딩을 신경망이 사용하기 좋은 형태로 변환
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        """
        t: [batch_size]
           각 데이터가 현재 몇 번째 Diffusion 단계인지 나타낸다.
        """

        half_dim = self.dim // 2

        # 서로 다른 주파수의 sin, cos를 이용해 시간 정보를 표현
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half_dim,
                device=t.device,
                dtype=torch.float32,
            )
            / max(half_dim - 1, 1)
        )

        # [batch_size, half_dim]
        angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)

        # [batch_size, dim]
        embedding = torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=1,
        )

        # dim이 홀수인 경우 차원을 하나 맞춘다.
        if embedding.shape[1] < self.dim:
            embedding = F.pad(
                embedding,
                (0, self.dim - embedding.shape[1]),
            )

        return self.mlp(embedding)


# ============================================================
# 2. 노이즈 예측 모델
# ============================================================
class NoisePredictor(nn.Module):
    """
    입력:
        x_t : 노이즈가 섞인 Mel-spectrogram
        t   : 현재 Diffusion 시간 단계

    출력:
        predicted_noise : 모델이 예측한 노이즈

    이 모델은 원본 음성 자체가 아니라,
    입력에 섞인 노이즈를 예측하도록 학습된다.
    """

    def __init__(self, in_channels=1, hidden_channels=64, time_dim=128):
        super().__init__()

        self.time_embedding = TimeEmbedding(time_dim)

        # 입력 Mel-spectrogram의 특징을 추출
        self.input_conv = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )

        # 시간 임베딩을 특징 맵에 더하기 위한 변환
        self.time_projection = nn.Linear(
            time_dim,
            hidden_channels,
        )

        # 주변 시간 프레임과 Mel 주파수의 특징을 학습
        self.block1 = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )

        # 입력과 중간 특징을 결합하는 잔차 연결
        self.block3 = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )

        # 최종적으로 입력과 같은 크기의 노이즈를 예측
        self.output_conv = nn.Conv2d(
            hidden_channels,
            in_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x_t, t):
        """
        x_t: [B, 1, mel_bins, time_frames]
        t:   [B]
        """

        # 시간 단계 t를 벡터로 변환
        t_embedding = self.time_embedding(t)

        # Mel-spectrogram의 특징 추출
        h = self.input_conv(x_t)

        # 시간 임베딩을 각 위치의 특징에 더한다.
        time_features = self.time_projection(t_embedding)
        time_features = time_features[:, :, None, None]

        h = h + time_features

        # 합성곱으로 특징을 학습
        h = self.block1(h)

        # 잔차 연결: 기존 특징을 보존하면서 추가 학습
        residual = h
        h = self.block2(h)
        h = h + residual

        h = self.block3(h)

        # 예측 노이즈 반환
        predicted_noise = self.output_conv(h)

        return predicted_noise


# ============================================================
# 3. Diffusion 모델
# ============================================================
class DiffusionModel(nn.Module):
    """
    NoisePredictor와 정방향/역방향 Diffusion을 결합한다.

    정방향:
        x_0 -> x_t
        원본에 노이즈를 추가한다.

    역방향:
        x_t -> x_(t-1) -> ... -> x_0
        모델이 예측한 노이즈를 이용해 복원한다.
    """

    def __init__(
        self,
        in_channels=1,
        hidden_channels=64,
        time_dim=128,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
    ):
        super().__init__()

        self.timesteps = timesteps

        # 노이즈 예측 신경망
        self.noise_predictor = NoisePredictor(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            time_dim=time_dim,
        )

        # ----------------------------------------------------
        # Diffusion 노이즈 스케줄
        # ----------------------------------------------------

        # 각 단계에서 추가되는 노이즈의 비율
        betas = torch.linspace(
            beta_start,
            beta_end,
            timesteps,
        )

        # 원본 정보가 남는 비율
        alphas = 1.0 - betas

        # 0단계부터 현재 단계까지 alpha를 누적 곱한다.
        alpha_bars = torch.cumprod(alphas, dim=0)

        # 아래 값들은 학습되는 파라미터가 아니라
        # Diffusion 계산에 사용하는 고정값이다.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        self.register_buffer(
            "sqrt_alpha_bars",
            torch.sqrt(alpha_bars),
        )

        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt(1.0 - alpha_bars),
        )

        # 역방향 과정에서 사용하는 분산
        alpha_bars_previous = F.pad(
            alpha_bars[:-1],
            (1, 0),
            value=1.0,
        )

        posterior_variance = (
            betas
            * (1.0 - alpha_bars_previous)
            / (1.0 - alpha_bars)
        )

        self.register_buffer(
            "posterior_variance",
            posterior_variance.clamp(min=1e-20),
        )

    # ========================================================
    # 4. 정방향 확산: 원본에 노이즈 추가
    # ========================================================
    def add_noise(self, x_0, t, noise=None):
        """
        x_0: 원본 Mel-spectrogram
        t:   각 샘플의 Diffusion 단계

        반환:
            x_t: t 단계만큼 노이즈가 추가된 데이터
            noise: 실제로 추가한 노이즈
        """

        if noise is None:
            noise = torch.randn_like(x_0)

        # 배치마다 다른 t를 적용할 수 있도록 크기를 맞춘다.
        sqrt_alpha_bar = self.sqrt_alpha_bars[t]
        sqrt_one_minus_alpha_bar = (
            self.sqrt_one_minus_alpha_bars[t]
        )

        sqrt_alpha_bar = sqrt_alpha_bar[:, None, None, None]
        sqrt_one_minus_alpha_bar = (
            sqrt_one_minus_alpha_bar[:, None, None, None]
        )

        # 원본 정보 + 노이즈
        x_t = (
            sqrt_alpha_bar * x_0
            + sqrt_one_minus_alpha_bar * noise
        )

        return x_t, noise

    # ========================================================
    # 5. 학습용 노이즈 예측
    # ========================================================
    def predict_noise(self, x_t, t):
        """
        노이즈가 섞인 x_t와 시간 t를 입력받아
        모델이 추정한 노이즈를 반환한다.
        """

        return self.noise_predictor(x_t, t)

    # ========================================================
    # 6. 역방향 확산의 한 단계
    # ========================================================
    @torch.no_grad()
    def reverse_step(self, x_t, t):
        """
        x_t에서 x_(t-1)로 한 단계 복원한다.

        t는 배치의 모든 샘플에 동일하게 적용되는 정수다.
        """

        batch_size = x_t.shape[0]

        # 현재 단계 t를 배치 크기에 맞춘다.
        t_batch = torch.full(
            (batch_size,),
            t,
            device=x_t.device,
            dtype=torch.long,
        )

        # 현재 x_t에 포함된 노이즈를 예측
        predicted_noise = self.predict_noise(x_t, t_batch)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]

        # DDPM 역방향 평균:
        # 예측한 노이즈를 이용해 x_t에서 노이즈 성분을 제거한다.
        mean = (
            x_t
            - (beta_t / torch.sqrt(1.0 - alpha_bar_t))
            * predicted_noise
        ) / torch.sqrt(alpha_t)

        # 마지막 단계에서는 추가 노이즈를 넣지 않는다.
        if t == 0:
            return mean

        # 그 외 단계에서는 정해진 분산에 따라 노이즈를 추가한다.
        variance = self.posterior_variance[t]
        
        if self.training :
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance)*noise

        return mean

    # ========================================================
    # 7. 전체 역방향 확산
    # ========================================================
    @torch.no_grad()
    def reverse_diffusion(self, x_t, start_t=None):
        """
        역방향 과정을 반복해 입력 x_t를 복원한다.

        x_t:
            역방향 복원을 시작할 Mel-spectrogram

        start_t:
            역방향을 시작할 시간 단계.
            지정하지 않으면 가장 마지막 단계에서 시작한다.

        반환:
            복원된 Mel-spectrogram
        """

        if start_t is None:
            start_t = self.timesteps - 1

        if not 0 <= start_t < self.timesteps:
            raise ValueError(
                f"start_t는 0부터 {self.timesteps - 1} "
                "사이여야 합니다."
            )

        x = x_t

        # 예: start_t=999라면
        # 999 -> 998 -> 997 -> ... -> 1 -> 0 순서로 복원
        for t in reversed(range(start_t + 1)):
            x = self.reverse_step(x, t)

        return x

    # ========================================================
    # 8. 원본 음성을 특정 단계까지 노이즈화한 뒤 복원
    # ========================================================
    @torch.no_grad()
    def reconstruct(self, x_0, start_t=200):
        """
        입력 Mel-spectrogram을 특정 단계까지 노이즈화한 다음,
        역방향 Diffusion으로 복원한다.

        주의:
            이 함수는 입력을 그대로 복사하는 것이 아니다.
            입력에 새 노이즈를 추가한 뒤 모델이 복원한다.

            start_t가 클수록 노이즈가 많이 추가된다.
        """

        if not 0 <= start_t < self.timesteps:
            raise ValueError(
                f"start_t는 0부터 {self.timesteps - 1} "
                "사이여야 합니다."
            )

        batch_size = x_0.shape[0]

        t = torch.full(
            (batch_size,),
            start_t,
            device=x_0.device,
            dtype=torch.long,
        )

        # 원본에 노이즈를 추가해 x_t 생성
        x_t, _ = self.add_noise(x_0, t)

        # x_t에서 역방향으로 복원
        reconstructed = self.reverse_diffusion(
            x_t,
            start_t=start_t,
        )

        return reconstructed