# this file is not runned yet it was made cause the chat was temp 

# run this first 
#pip install ultralytics opencv-python pyyaml tqdm

import os
import cv2
import yaml
import zipfile
from tqdm import tqdm # type: ignore
from ultralytics import YOLO

# -----------------------
# USER SETTINGS
# -----------------------

INPUT_MEDIA = "input_media"
DATASET_DIR = "dataset"
MODEL_PATH = "yolo26s.pt"

FRAME_SKIP = 15
CONF_THRESHOLD = 0.5

CLASSES = ["person", "cat", "dog", "cow"]

# -----------------------
# CREATE DATASET FOLDERS
# -----------------------

images_dir = os.path.join(DATASET_DIR, "images", "val")
labels_dir = os.path.join(DATASET_DIR, "labels", "val")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# -----------------------
# LOAD MODEL
# -----------------------

print("\nLoading YOLO model...")
model = YOLO(MODEL_PATH)

img_count = 0

# -----------------------
# PROCESS MEDIA
# -----------------------

files = os.listdir(INPUT_MEDIA)

for file in files:

    path = os.path.join(INPUT_MEDIA, file)

    if file.lower().endswith((".mp4",".avi",".mov")):

        print(f"\nProcessing video: {file}")

        cap = cv2.VideoCapture(path)
        frame_id = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_id % FRAME_SKIP == 0:

                img_name = f"frame_{img_count}.jpg"
                img_path = os.path.join(images_dir, img_name)

                cv2.imwrite(img_path, frame)

                results = model(frame)

                label_path = os.path.join(labels_dir, img_name.replace(".jpg",".txt"))

                with open(label_path,"w") as f:

                    for r in results:

                        if r.boxes is None:
                            continue

                        for box in r.boxes:

                            conf = float(box.conf)

                            if conf < CONF_THRESHOLD:
                                continue

                            cls = int(box.cls)
                            name = model.names[cls]

                            if name not in CLASSES:
                                continue

                            x,y,w,h = box.xywhn[0].tolist()

                            f.write(f"{cls} {x} {y} {w} {h}\n")

                img_count += 1

            frame_id += 1

        cap.release()

    elif file.lower().endswith((".jpg",".png",".jpeg")):

        print(f"\nProcessing image: {file}")

        frame = cv2.imread(path)

        img_name = f"frame_{img_count}.jpg"
        img_path = os.path.join(images_dir, img_name)

        cv2.imwrite(img_path, frame)

        results = model(frame)

        label_path = os.path.join(labels_dir, img_name.replace(".jpg",".txt"))

        with open(label_path,"w") as f:

            for r in results:

                if r.boxes is None:
                    continue

                for box in r.boxes:

                    conf = float(box.conf)

                    if conf < CONF_THRESHOLD:
                        continue

                    cls = int(box.cls)
                    name = model.names[cls]

                    if name not in CLASSES:
                        continue

                    x,y,w,h = box.xywhn[0].tolist()

                    f.write(f"{cls} {x} {y} {w} {h}\n")

        img_count += 1

print("\nFrame extraction + auto labeling complete.")

# -----------------------
# CREATE data.yaml
# -----------------------

data_yaml = {
    "path": DATASET_DIR,
    "val": "images/val",
    "names": CLASSES
}

yaml_path = os.path.join(DATASET_DIR, "data.yaml")

with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f)

print("data.yaml created.")

# -----------------------
# ZIP DATASET
# -----------------------

zip_name = "dataset.zip"

print("\nCreating zip file...")

with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:

    for root, dirs, files in os.walk(DATASET_DIR):

        for file in files:

            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, DATASET_DIR)

            zipf.write(file_path, arcname)

print("Dataset zipped successfully.")

# -----------------------
# RUN ACCURACY TEST
# -----------------------

print("\nRunning model evaluation...\n")

metrics = model.val(data=yaml_path)

print("\nMODEL PERFORMANCE\n")

print(f"Precision: {metrics.box.mp*100:.2f}%")
print(f"Recall: {metrics.box.mr*100:.2f}%")
print(f"mAP50: {metrics.box.map50*100:.2f}%")
print(f"mAP50-95: {metrics.box.map*100:.2f}%")

print("\nAll done.")