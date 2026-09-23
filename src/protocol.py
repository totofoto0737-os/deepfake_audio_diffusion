from pathlib import Path


def read_labels(root): #라벨 읽기
    root = Path(root)

    # CM 프로토콜 파일만 사용
    key_file = root / "keys" / "LA" / "CM" / "trial_metadata.txt"

    if not key_file.exists():
        raise FileNotFoundError(
            f"CM trial_metadata.txt not found: {key_file}"
        )

    labels = {}

    for line in key_file.read_text(
        encoding="utf-8",
        errors="ignore"
    ).splitlines():

        a = line.split()

        # CM 형식은 최소 6개 항목
        if len(a) < 6:
            continue

        # 오디오 파일 ID
        file_id = a[1]

        # CM 라벨은 a[5]
        label_word = a[5].lower()

        if label_word == "bonafide":
            label = 0

        elif label_word == "spoof":
            label = 1

        else:
            continue

        labels[file_id] = label

    print("읽은 라벨 수:", len(labels))
    print("라벨 예시:", list(labels.items())[:5])

    return labels