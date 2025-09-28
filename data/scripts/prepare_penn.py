import glob
import os

import cv2
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm


dataset_root_path = ""


frames_dir = os.path.join(dataset_root_path, "frames")
labels_dir = os.path.join(dataset_root_path, "labels")
output_videos_dir = os.path.join(dataset_root_path, "videos_mp4")
output_csv_path = os.path.join(dataset_root_path, "annotations.csv")

os.makedirs(output_videos_dir, exist_ok=True)

print(f"[*] Original frames: {frames_dir}")
print(f"[*] Original labels: {labels_dir}")
print(f"[*] Folder for MP4 videos: {output_videos_dir}")
print(f"[*] Output CSV file: {output_csv_path}")

label_files = sorted(glob.glob(os.path.join(labels_dir, "*.mat")))

csv_data = []
DEFAULT_FPS = 30

print(f"\n[*] Starting processing {len(label_files)} videos...")

for label_file_path in tqdm(label_files, desc="Building videos"):
    base_name = os.path.splitext(os.path.basename(label_file_path))[0]

    mat_data = loadmat(label_file_path)
    action_label = mat_data["action"][0]

    current_frames_dir = os.path.join(frames_dir, base_name)
    frame_files = sorted(glob.glob(os.path.join(current_frames_dir, "*.jpg")))

    if not frame_files:
        print(f"\n[!] Warning: for video {base_name} no frames found. Skipping.")
        continue

    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape

    output_video_path = os.path.join(output_videos_dir, f"{base_name}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, DEFAULT_FPS, (width, height))

    for frame_path in frame_files:
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()

    csv_data.append({"video_path": output_video_path, "label": action_label})

print("\n[*] Creating and saving CSV file...")
df = pd.DataFrame(csv_data)
df.to_csv(output_csv_path, index=False, columns=["video_path", "label"])

print(f"\nDone! Results saved:")
print(f"    - Videos: {output_videos_dir}")
print(f"    - CSV file: {output_csv_path}")
