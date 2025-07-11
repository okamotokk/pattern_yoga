#LLMs[1][2] are used to create this code.
#[1] OpenAI. GPT-4o. (2025). Open AI. [Online]. Available: https://chatgpt.com/?model=gpt-4o
#[2] GitHub. GitHub Copilot. (2025). GitHub. [Online].
import cv2
import mediapipe as mp
import json
import os
from glob import glob

# 初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# 読み込む画像があるディレクトリ
input_dir = "yoga\kakato"  # ←ここを適宜変更
image_paths = glob(os.path.join(input_dir, "*.jpg")) + glob(os.path.join(input_dir, "*.png"))

# 特定ランドマークのみ（例：両手首）
target_landmarks = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER,mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX, mp_pose.PoseLandmark.RIGHT_INDEX, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

# すべての結果を格納するリスト
results_list = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f"読み込み失敗: {image_path}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print(f"ランドマーク検出失敗: {image_path}")
        continue

    landmarks = results.pose_landmarks.landmark
    landmark_data = {}

    for lm_enum in target_landmarks:
        lm = landmarks[lm_enum.value]
        landmark_data[lm_enum.name] = {
            "x": round(lm.x, 4),
            "y": round(lm.y, 4),
            "z": round(lm.z, 4),
            "visibility": round(lm.visibility, 3)
        }

    # 個々の画像の結果として格納
    results_list.append({
        "image_path": image_path,
        "landmarks": landmark_data
    })

# JSONファイルに保存
with open("output_kakato.json", "w", encoding="utf-8") as f:
    json.dump(results_list, f, indent=2, ensure_ascii=False)

print("すべての画像を処理し、JSONファイルに保存しました。→ output_kakato.json")
