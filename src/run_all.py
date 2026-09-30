# # ============================================================
# # 필요한 라이브러리 가져오기
# # ============================================================

# import argparse
# import random

# from pathlib import Path

# import torch
# from torch.utils.data import DataLoader

# from protocol import read_labels
# from dataset import AudioDataset
# from model import DiffusionModel
# from config import *
# from evaluate import load_model, evaluate_model


# # ============================================================
# # 1. 오디오 파일 찾기
# # ============================================================

# def get_audio_files(root):

#     """
#     지정한 폴더 안에서
#     .flac / .wav / .mp3 파일을 모두 찾는다.
#     """

#     root = Path(root)

#     audio_files = []

#     # root 폴더 안의 모든 파일을 확인
#     for path in root.rglob("*"):

#         # 파일 확장자 확인
#         extension = path.suffix.lower()

#         # 음성 파일이면 저장
#         if extension in [".flac", ".wav", ".mp3"]:
#             audio_files.append(path)

#     # 파일 이름 순서대로 정렬
#     audio_files.sort()

#     return audio_files





# # ============================================================
# # 3. 프로그램 시작 -> 명령어로 실행시킬 때 시작
# # ============================================================

# def main():

#     # ========================================================
#     # 실행할 때 입력받을 옵션 설정-> 명령어로 넣어준 옵션대로 실행하도록 만듬
#     # ========================================================

#     parser = argparse.ArgumentParser()


#     # 오디오 데이터 폴더를 "--audio-root" 명령어로 받는다.
#     parser.add_argument(
#         "--audio-root",
#         required=True
#     )


#     # 라벨 파일이 있는 폴더의 정보를 "--key-root" 명령어로 받는다. 
#     parser.add_argument(
#         "--key-root",
#         required=True
#     )


#     # 결과를 저장할 폴더경로를 "--out-dir" 명령어로 받는다.
#     parser.add_argument(
#         "--out-dir",
#         default="outputs/diffusion_run"
#     )


#     # 학습 횟수를 "--epochs" 명령어로 받는다. 
#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=10
#     )


#     # 한 번에 학습할 데이터 개수를 "--batch-size" 명령어로 받는다. 
#     parser.add_argument(
#         "--batch-size",
#         type=int,
#         default=BATCH_SIZE
#     )


#     # 학습에 사용할 REAL 데이터 수를 "--max-train-real" 명령어로 받는다. 
#     parser.add_argument(
#         "--max-train-real",
#         type=int,
#         default=1000
#     )


#     # Validation REAL 데이터 수를 "--max-val-real" 명령어로 받는다. 
#     parser.add_argument(
#         "--max-val-real",
#         type=int,
#         default=200
#     )


#     # Validation FAKE 데이터 수를 "--max-val-real" 명령어로 받는다. 
#     parser.add_argument(
#         "--max-val-fake",
#         type=int,
#         default=200
#     )


#     # Test REAL 데이터 수를 "--max-test-real" 명령어로 받는다. 
#     parser.add_argument(
#         "--max-test-real",
#         type=int,
#         default=200
#     )


#     # Test FAKE 데이터 수를 "--max-test-fake" 명령어로 받는다. 
#     parser.add_argument(
#         "--max-test-fake",
#         type=int,
#         default=200
#     )


#     # 실제 명령어에서 입력한 값을 저장한다. : '--epochs' 10 -> args.epochs = 10
#     args = parser.parse_args()


#     # ========================================================
#     # 4. 랜덤 고정
#     # ========================================================

#     # 실행할 때마다 비슷한 데이터를 뽑도록
#     # 랜덤값을 고정한다.
#     random.seed(42)


#     # ========================================================
#     # 5. 라벨 읽기
#     # ========================================================

#     labels = read_labels(
#         args.key_root
#     )


#     # ========================================================
#     # 6. 오디오 파일 찾기
#     # ========================================================

#     audio_files = get_audio_files(
#         args.audio_root
#     )


#     # ========================================================
#     # 7. 오디오와 라벨 연결
#     # ========================================================

#     all_data = []

#     for path in audio_files:

#         # 파일 이름에서 확장자를 제거
#         # ex) LA_E_9332881.flac -> LA_E_9332881
#         file_id = path.stem


#         # 라벨 파일에 해당 파일이 있는지 확인
#         if file_id in labels:

#             label = labels[file_id]

#             # (파일 경로, 라벨) 형태로 저장
#             all_data.append(
#                 (path, label)
#             )


#     # ========================================================
#     # 8. 데이터 순서를 섞음
#     # ========================================================

#     random.shuffle(
#         all_data
#     )


#     # ========================================================
#     # 9. REAL / FAKE 분리
#     # ========================================================

#     real_data = []
#     fake_data = []


#     for item in all_data:

#         path, label = item

#         if label == 0:

#             # REAL
#             real_data.append(item)

#         elif label == 1:

#             # FAKE
#             fake_data.append(item)


#     # ========================================================
#     # 10. Train / Validation / Test 데이터 만들기
#     # ========================================================

#     # --------------------------------------------------------
#     # Train
#     # --------------------------------------------------------
#     #
#     train_data = real_data[
#         :args.max_train_real
#     ]


#     # --------------------------------------------------------
#     # Validation REAL
#     # --------------------------------------------------------

#     val_real = real_data[
#         args.max_train_real:
#         args.max_train_real + args.max_val_real
#     ]


#     # --------------------------------------------------------
#     # Validation FAKE
#     # --------------------------------------------------------

#     val_fake = fake_data[
#         :args.max_val_fake
#     ]


#     # --------------------------------------------------------
#     # Test REAL
#     # --------------------------------------------------------

#     test_real = real_data[
#         args.max_train_real + args.max_val_real:
#         args.max_train_real
#         + args.max_val_real
#         + args.max_test_real
#     ]


#     # --------------------------------------------------------
#     # Test FAKE
#     # --------------------------------------------------------

#     test_fake = fake_data[
#         args.max_val_fake:
#         args.max_val_fake
#         + args.max_test_fake
#     ]


#     # Test에는 REAL + FAKE 둘 다 들어간다.
#     test_data = test_real + test_fake


#     # ========================================================
#     # 11. 사용할 장치 결정
#     # ========================================================

#     if torch.cuda.is_available():

#         device = torch.device("cuda")

#     else:

#         device = torch.device("cpu")


#     print(
#         "Device:",
#         device
#     )

#     print(
#         "Train REAL:",
#         len(train_data)
#     )

#     print(
#         "Validation:",
#         len(val_real) + len(val_fake)
#     )

#     print(
#         "Test:",
#         len(test_data)
#     )


#     # ========================================================
#     # 12. DataLoader 만들기
#     # ========================================================

#     train_loader = DataLoader(
#         AudioDataset(train_data),

#         batch_size=args.batch_size,

#         # 학습할 때 데이터 순서를 섞음
#         shuffle=True,

#         # Windows에서 안전하게 사용
#         num_workers=0
#     )
    
#     # ========================================================
#     # Validation / Test DataLoader 만들기
#     # ========================================================

#     # 검증 데이터: REAL + FAKE
#     val_data = val_real + val_fake

#     val_loader = DataLoader(
#         AudioDataset(val_data),
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=0
#     )

    
#     # 테스트 데이터: REAL + FAKE
#     test_loader = DataLoader(
#         AudioDataset(test_data),
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=0
#     )


#     # ========================================================
#     # 13. Diffusion 모델 만들기
#     # ========================================================

#     model = DiffusionModel(
#         in_channels=IN_CHANNELS,
#         hidden_channels=HIDDEN_CHANNELS,
#         time_dim=TIME_DIM,
#         timesteps=TIMESTEPS,
#     )

#     model = model.to(device)


#     # ========================================================
#     # 14. Optimizer 만들기
#     # ========================================================

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=LR
#     )


#     # ========================================================
#     # 15. 실제 학습 시작
#     # ========================================================

#     for epoch in range(args.epochs):

#         # 학습 모드
#         model.train()

#         total_loss = 0


#         # ----------------------------------------------------
#         # Train 데이터 하나씩 가져오기
#         # ----------------------------------------------------

#         for x, label, path in train_loader:

#             # Mel-spectrogram을 CPU/GPU로 이동
#             x = x.to(device)


#             # ------------------------------------------------
#             # 랜덤 timestep 선택
#             # ------------------------------------------------
            
#             # 각 데이터에 적용할 Diffusion 시간 단계를 무작위로 선택
#             t = torch.randint(
#                 0,
#                 TIMESTEPS,
#                 (x.size(0),),
#                 device=device
#             )


#             # ------------------------------------------------
#             # 2. 원본 Mel-spectrogram에 노이즈 추가
#             # ------------------------------------------------

#             # model.py에 구현된 노이즈 스케줄을 사용한다.
#             # noisy_x: 노이즈가 추가된 Mel-spectrogram
#             # noise: 실제로 추가한 노이즈
#             noisy_x, noise = model.add_noise(
#                 x,
#                 t
#             )


#             # ------------------------------------------------
#             # 3. 모델이 노이즈 예측
#             # ------------------------------------------------

#             # 모델이 noisy_x에 들어 있는 노이즈를 예측한다.
#             # t는 정수 timestep 그대로 전달한다.
#             predicted_noise = model.predict_noise(
#                 noisy_x,
#                 t
#             )


#             # ------------------------------------------------
#             # 4. 실제 노이즈와 예측 노이즈 비교
#             # ------------------------------------------------

#             loss = (
#                 predicted_noise - noise
#             ).pow(2).mean()

#             # ------------------------------------------------
#             # 실제 noise와 예측 noise 비교
#             # ------------------------------------------------

#             loss = (
#                 predicted_noise - noise
#             ).pow(2).mean()


#             # ------------------------------------------------
#             # 이전 gradient 초기화
#             # ------------------------------------------------

#             optimizer.zero_grad()


#             # ------------------------------------------------
#             # 오차를 이용해서 역전파
#             # ------------------------------------------------

#             loss.backward()


#             # ------------------------------------------------
#             # 너무 큰 gradient 방지
#             # ------------------------------------------------

#             torch.nn.utils.clip_grad_norm_(
#                 model.parameters(),
#                 1
#             )


#             # ------------------------------------------------
#             # 모델의 가중치 업데이트
#             # ------------------------------------------------

#             optimizer.step()


#             # loss 누적
#             total_loss += loss.item()


#         # ----------------------------------------------------
#         # Epoch 결과 출력
#         # ----------------------------------------------------

#         average_loss = (
#             total_loss /
#             max(1, len(train_loader))
#         )


#         print(
#             f"Epoch {epoch + 1}/{args.epochs} "
#             f"loss={average_loss:.6f}"
#         )



#     # ========================================================
#     # 16. 학습된 모델 저장
#     # ========================================================

#     # 결과 저장 폴더
#     output_dir = Path(args.out_dir)

#     output_dir.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     # 학습된 모델의 가중치 저장
#     checkpoint_path = output_dir / "model.pt"

    
    
#     torch.save(
#         model.state_dict(),
#         checkpoint_path
#     )

#     print(f"모델 저장 완료: {checkpoint_path}")


    

#     #==================================
#     #19 threshold 계산 
#     #=====================================
#     threshold = float(
#     np.percentile(
#         real_scores,
#         REAL_PERCENTILE
#         )
#     )

#     # ========================================================
#     # 20. Threshold로 REAL / FAKE 판단
#     # ========================================================

#     # score가 threshold보다 크면
#     # → FAKE = 1
#     #
#     # score가 threshold보다 작거나 같으면
#     # → REAL = 0

    


#     # ========================================================
#     # 22. 결과 저장 폴더 만들기
#     # ========================================================

#     output_dir = Path(
#         args.out_dir
#     )

#     output_dir.mkdir(
#         parents=True,
#         exist_ok=True
#     )


#     # ========================================================
#     # 23. 학습된 모델 저장
#     # ========================================================

#     torch.save(
#         model.state_dict(),
#         output_dir / "model.pt"
#     )


#     # ========================================================
#     # 24. Test 결과 CSV 저장
#     # ========================================================

#     test_result.to_csv(
#         output_dir / "scores.csv",
#         index=False,
#         encoding="utf-8-sig"
#     )


#     # ========================================================
#     # 25. Threshold 정보 저장
#     # ========================================================

#     threshold_info = {
#         "threshold": threshold,
#         "percentile": REAL_PERCENTILE
#     }


#     with open(
#         output_dir / "threshold.json",
#         "w"
#     ) as file:

#         json.dump(
#             threshold_info,
#             file,
#             indent=2
#         )


#     # ========================================================
#     # 26. 결과 출력
#     # ========================================================

#     print(
#         "Threshold:",
#         threshold
#     )

#     print(
#         "Saved:",
#         output_dir
#     )
    



# # ============================================================
# # 이 파일을 직접 실행했을 때만 main() 실행 ->다른 파일에서 run_all파일 함수나 변수를 import해도 또 실행하지 않도록 조치
# # ============================================================

# if __name__ == "__main__":

#     main()