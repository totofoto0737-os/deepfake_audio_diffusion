from torch.utils.data import Dataset
from preprocess import audio_to_mel


# -----------------------------------------
# 오디오 데이터를 불러오는 Dataset
# -----------------------------------------
class AudioDataset(Dataset):

    def __init__(self, items):
        """
        items에는 다음과 같은 형태의 데이터가 들어있음.

        [
            (오디오파일경로, 라벨),
            (오디오파일경로, 라벨),
            ...
        ]

        예:
        ("data/real/audio1.flac", 0)
        ("data/fake/audio2.flac", 1)
        """

        # 전달받은 데이터를 저장
        self.items = items


    def __len__(self):
        """
        데이터가 몇 개인지 알려주는 함수
        """

        return len(self.items)


    def __getitem__(self, index):
        """
        index 번째 데이터를 하나 가져오는 함수
        """

        # index 번째 데이터에서
        # 파일 경로와 라벨을 꺼냄
        path, label = self.items[index]

        # 오디오 파일을
        # Mel-spectrogram으로 변환
        mel = audio_to_mel(path)

        # Dataset에서 최종적으로 반환할 값
        #
        # mel   : Mel-spectrogram
        # label : REAL(0) / FAKE(1)
        # path  : 원본 오디오 파일 경로
        return mel, label, str(path)