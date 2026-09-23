import torch
import torch.nn as nn


# -----------------------------------------
# 기본 블록
# -----------------------------------------
class Block(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()

        # 첫 번째 합성곱
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1
        )

        # 데이터의 분포를 안정적으로 만들어줌
        self.norm1 = nn.GroupNorm(
            num_groups=4,
            num_channels=output_channels
        )

        # 활성화 함수
        self.act1 = nn.SiLU()

        # 두 번째 합성곱
        self.conv2 = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=1
        )

        # 두 번째 정규화
        self.norm2 = nn.GroupNorm(
            num_groups=4,
            num_channels=output_channels
        )

        # 두 번째 활성화 함수
        self.act2 = nn.SiLU()


    def forward(self, x):

        # 입력 → 첫 번째 Conv
        x = self.conv1(x)

        # 정규화
        x = self.norm1(x)

        # 활성화
        x = self.act1(x)

        # 두 번째 Conv
        x = self.conv2(x)

        # 정규화
        x = self.norm2(x)

        # 활성화
        x = self.act2(x)

        return x


# -----------------------------------------
# 아주 작은 U-Net
# -----------------------------------------
class TinyUNet(nn.Module):

    def __init__(self, base=32):
        super().__init__()

        # ---------------------------------
        # 1. 처음 특징 추출
        # ---------------------------------
        # 입력:
        #   채널 1개 (Mel-spectrogram)
        #
        # 출력:
        #   특징 32개
        self.first_block = Block(
            input_channels=1,
            output_channels=base
        )


        # ---------------------------------
        # 2. 크기를 줄이면서 특징을 늘림
        # ---------------------------------
        # 32 → 64 채널
        # 이미지 크기는 절반으로 줄어듦
        self.down = nn.Conv2d(
            in_channels=base,
            out_channels=base * 2,
            kernel_size=4,
            stride=2,
            padding=1
        )


        # ---------------------------------
        # 3. 가운데 부분
        # ---------------------------------
        # 64채널을 다시 처리
        self.middle = Block(
            input_channels=base * 2,
            output_channels=base * 2
        )


        # ---------------------------------
        # 4. 크기를 다시 키움
        # ---------------------------------
        # 64 → 32 채널
        # 이미지 크기는 다시 2배
        self.up = nn.ConvTranspose2d(
            in_channels=base * 2,
            out_channels=base,
            kernel_size=4,
            stride=2,
            padding=1
        )


        # ---------------------------------
        # 5. 최종 출력
        # ---------------------------------
        # up 결과와 처음 특징을 합친 후
        # 최종적으로 채널 1개를 출력
        self.output = nn.Conv2d(
            in_channels=base * 2,
            out_channels=1,
            kernel_size=1
        )


    def forward(self, x, t):

        # 1. 처음 특징 추출
        h = self.first_block(x)

        # 2. 크기를 줄임
        z = self.down(h)

        # 3. 가운데에서 특징 처리
        z = self.middle(z)

        # 4. 크기를 다시 키움
        u = self.up(z)

        # 5. 처음에 얻었던 특징 h와 합침
        # 채널 방향(dim=1)으로 연결
        combined = torch.cat(
            [u, h],
            dim=1
        )

        # 6. 최종 출력
        output = self.output(combined)

        return output