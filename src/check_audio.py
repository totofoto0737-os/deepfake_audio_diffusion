from pathlib import Path
import random
import soundfile as sf

from protocol import read_labels


# -----------------------------------------
# 경로 설정
# -----------------------------------------
audio_root = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "ASVspoof2021_LA_eval"
)

key_root = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "LA-keys-full"
)


# -----------------------------------------
# 라벨 읽기
# -----------------------------------------
labels = read_labels(key_root)


# -----------------------------------------
# 오디오 파일 찾기
# -----------------------------------------
files = sorted(
    p for p in audio_root.rglob("*")
    if p.suffix.lower() in (".flac", ".wav", ".mp3")
)


# -----------------------------------------
# 파일 + 라벨 연결
# -----------------------------------------
all_data = [
    (p, labels[p.stem])
    for p in files
    if p.stem in labels
]


# run_all.py와 똑같이 랜덤 섞기
random.seed(42)
random.shuffle(all_data)


# -----------------------------------------
# REAL만 뽑기
# label 0 = REAL
# -----------------------------------------
real_data = [
    (p, y)
    for p, y in all_data
    if y == 0
]


# 실제 학습에 쓰는 앞 1000개 REAL
train_data = real_data[:1000]

print("검사할 학습 REAL 파일 수:", len(train_data))


# -----------------------------------------
# 실제로 파일 전체를 읽어보기
# -----------------------------------------
bad_count = 0

for i, (path, label) in enumerate(train_data, start=1):

    try:
        # 중요:
        # sf.info()가 아니라 실제 오디오 데이터를 끝까지 읽음
        data, sr = sf.read(str(path))

    except Exception as e:

        bad_count += 1

        print()
        print("==============================")
        print(f"[오류 {bad_count}]")
        print("순서:", i)
        print("파일:", path)
        print("원인:", e)
        print("==============================")


print()
print("==============================")
print("검사 완료")
print("검사한 파일:", len(train_data))
print("읽기 실패 파일:", bad_count)
print("==============================")