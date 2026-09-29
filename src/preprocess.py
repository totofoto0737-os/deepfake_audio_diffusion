import numpy as np
import librosa
import torch
import torch.nn.functional as F
import subprocess

from config import (
    SR,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    IMAGE_SIZE
)


# ==================================================
# 오디오 파일을 Mel-spectrogram으로 변환하는 함수
# ==================================================
def audio_to_mel(path):

    # ----------------------------------------------
    # 1. FFmpeg를 사용해 오디오 파일 읽기
    # ----------------------------------------------
    # 기존 librosa.load() 대신 FFmpeg를 사용한다.
    #
    # 기존 문제:
    # soundfile에서 다음과 같은 오류가 발생할 수 있다.
    # LibsndfileError: flac decoder lost sync
    #
    # 해결:
    # FFmpeg로 오디오를 읽고,
    # NumPy 배열로 변환한다.
    #
    # -i : 입력 오디오 파일
    # -f f32le : 32비트 실수형 오디오 데이터
    # -acodec pcm_f32le : 출력 오디오를 float32로 설정
    # -ac 1 : 모노(1채널)로 변환
    # -ar SR : SR 샘플레이트로 변환
    # pipe:1 : 파일로 저장하지 않고 메모리로 출력

    result = subprocess.run(
        [
            "ffmpeg",
            "-v", "error",
            "-i", str(path),
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", "1",
            "-ar", str(SR),
            "pipe:1"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )

    # FFmpeg가 출력한 바이트 데이터를
    # NumPy의 float32 배열로 변환한다.
    y = np.frombuffer(
        result.stdout,
        dtype=np.float32
    )

    # FFmpeg에서 이미 SR로 변환했으므로
    # 현재 샘플레이트는 SR이다.
    sr = SR


    # ----------------------------------------------
    # 2. 빈 오디오인지 확인
    # ----------------------------------------------
    # 음성 데이터의 길이가 0이면 오류 발생
    if len(y) == 0:
        raise ValueError(
            f"empty audio: {path}"
        )


    # ----------------------------------------------
    # 3. Mel-spectrogram 만들기
    # ----------------------------------------------
    #
    # 음성 파형
    #    ↓
    # Mel-spectrogram
    #
    # y          : 오디오 파형
    # sr         : 샘플레이트
    # n_fft      : FFT 분석 구간 크기
    # hop_length : 프레임 사이 이동 간격
    # n_mels     : Mel 주파수 영역의 개수
    # power=2    : 파워 스펙트로그램 사용

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2
    )


    # ----------------------------------------------
    # 4. 값을 dB 단위로 변환
    # ----------------------------------------------
    # 파워 스펙트로그램을 dB 단위로 변환한다.
    #
    # ref=np.max
    # → 가장 큰 값을 기준으로 상대적인 dB를 계산한다.

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )


    # ----------------------------------------------
    # 5. 0 ~ 1 사이로 정규화
    # ----------------------------------------------
    #
    # 가장 작은 값 → 0
    # 가장 큰 값   → 1
    #
    # 1e-8은 분모가 0이 되는 것을 방지한다.

    mel = (
        mel - mel.min()
    ) / (
        mel.max() - mel.min() + 1e-8
    )


    # ----------------------------------------------
    # 6. NumPy → PyTorch Tensor
    # ----------------------------------------------
    #
    # 현재:
    #     mel = [주파수, 시간]
    #
    # Tensor로 변환하고 자료형을 float32로 설정한다.

    x = torch.tensor(
        mel,
        dtype=torch.float32
    )


    # ----------------------------------------------
    # 7. 채널 차원 추가
    # ----------------------------------------------
    #
    # [주파수, 시간]
    #       ↓
    # [1, 주파수, 시간]
    #
    # 1은 흑백 이미지처럼 채널이 1개라는 뜻이다.

    x = x.unsqueeze(0)


    # ----------------------------------------------
    # 8. 배치 차원 추가
    # ----------------------------------------------
    #
    # [1, 주파수, 시간]
    #       ↓
    # [1, 1, 주파수, 시간]
    #
    # PyTorch 이미지 입력 형태:
    # [배치, 채널, 높이, 너비]

    x = x.unsqueeze(0)


    # ----------------------------------------------
    # 9. 크기를 IMAGE_SIZE에 맞게 변경
    # ----------------------------------------------
    #
    # Mel-spectrogram의 시간 길이는
    # 오디오 길이에 따라 달라질 수 있다.
    #
    # 따라서 모델에 넣기 전에
    # 모든 데이터의 크기를 동일하게 맞춘다.
    #
    # 결과 크기:
    # [1, 1, N_MELS, IMAGE_SIZE]

    x = F.interpolate(
        x,
        size=(N_MELS, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )


    # ----------------------------------------------
    # 10. 배치 차원 제거
    # ----------------------------------------------
    #
    # 현재:
    # [1, 1, N_MELS, IMAGE_SIZE]
    #
    # Dataset에서 데이터 하나를 반환하므로
    # 배치 차원을 제거한다.
    #
    # 결과:
    # [1, N_MELS, IMAGE_SIZE]

    x = x.squeeze(0)


    # ----------------------------------------------
    # 11. 완성된 Mel-spectrogram 반환
    # ----------------------------------------------
    return x