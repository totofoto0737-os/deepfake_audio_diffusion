# Diffusion Deepfake Audio Detection

REAL/bonafide만 학습하고, validation REAL score의 95 percentile을 threshold로 사용합니다.

설치:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch librosa soundfile numpy pandas scikit-learn tqdm
```

실행:
```powershell
python src/run_all.py --audio-root "data\ASVspoof2021_LA_eval" --key-root "data\LA-keys-full" --out-dir "outputs\diffusion_run" --epochs 10 --max-train-real 1000 --max-val-real 200 --max-val-fake 200 --max-test-real 200 --max-test-fake 200
```
