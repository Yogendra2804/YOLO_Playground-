

#I ahven't run this yet but made it from an temp chat in chatGPT.
# RUn this before hand to install the packages ->
# pip install ultralytics opencv-python pyyaml

import os
import cv2 # type: ignore
import yaml # type: ignore
from ultralytics import YOLO # type: ignore

# -----------------------------
# USER SETTINGS (EDIT THESE)
# -----------------------------

INPUT_PATH = "input_media"      # folder containing videos or images
DATASET_PATH = "dataset"        # where dataset will be created
MODEL_PATH = "yolo26s.pt"

FRAME_SKIP = 15                 # take 1 frame every 15 frames
CONF_THRESHOLD = 0.4

CLASSES = ["person", "cat", "dog", "sheep"]

# -----------------------------
# CREATE FOLDERS
# -----------------------------

images_dir = os.path.join(DATASET_PATH, "images", "val")
labels_dir = os.path.join(DATASET_PATH, "labels", "val")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# -----------------------------
# LOAD MODEL
# -----------------------------

print("\nLoading YOLO model...")
model = YOLO(MODEL_PATH)

img_counter = 0

# -----------------------------
# PROCESS FILES
# -----------------------------

for file in os.listdir(INPUT_PATH):

    path = os.path.join(INPUT_PATH, file)

    # -------------------------
    # VIDEO FILE
    # -------------------------

    if file.lower().endswith((".mp4", ".avi", ".mov")):

        print(f"\nProcessing video: {file}")

        cap = cv2.VideoCapture(path)

        frame_id = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_id % FRAME_SKIP == 0:

                img_name = f"frame_{img_counter}.jpg"
                img_path = os.path.join(images_dir, img_name)

                cv2.imwrite(img_path, frame)

                results = model(frame)

                label_file = os.path.join(labels_dir, img_name.replace(".jpg",".txt"))

                with open(label_file,"w") as f:

                    for r in results:

                        if r.boxes is None:
                            continue

                        for box in r.boxes:

                            cls = int(box.cls)
                            name = model.names[cls]
                            conf = float(box.conf)

                            if name not in CLASSES:
                                continue

                            dataset_id = CLASSES.index(name)

                            if name not in CLASSES:
                                print("Ignoring:", name)
                                continue
                            
                            if conf < CONF_THRESHOLD:
                                continue

                            x,y,w,h = box.xywhn[0].tolist()

                            f.write(f"{cls} {x} {y} {w} {h}\n")

                img_counter += 1

            frame_id += 1

        cap.release()

    # -------------------------
    # IMAGE FILE
    # -------------------------

    elif file.lower().endswith((".jpg",".jpeg",".png")):

        print(f"\nProcessing image: {file}")

        frame = cv2.imread(path)

        img_name = f"frame_{img_counter}.jpg"
        img_path = os.path.join(images_dir, img_name)

        cv2.imwrite(img_path, frame)

        results = model(frame)

        label_file = os.path.join(labels_dir, img_name.replace(".jpg",".txt"))

        with open(label_file,"w") as f:

            for r in results:

                if r.boxes is None:
                    continue

                for box in r.boxes:

                    cls = int(box.cls)
                    conf = float(box.conf)

                    name = model.names[cls]

                    if name not in CLASSES:
                        continue

                    if conf < CONF_THRESHOLD:
                        continue

                    x,y,w,h = box.xywhn[0].tolist()

                    f.write(f"{cls} {x} {y} {w} {h}\n")

        img_counter += 1


print("\nDataset creation finished.")

# -----------------------------
# CREATE DATA.YAML
# -----------------------------

yaml_path = os.path.join(DATASET_PATH,"data.yaml")

data = {
    "path": DATASET_PATH,
    "train": "images/val",
    "val": "images/val",
    "names": {i: name for i, name in enumerate(CLASSES)}
}

with open(yaml_path,"w") as f:
    yaml.dump(data,f)

print("data.yaml created.")

# -----------------------------
# RUN ACCURACY EVALUATION
# -----------------------------

print("\nRunning accuracy evaluation...\n")

metrics = model.val(data=yaml_path)

print("\nMODEL PERFORMANCE\n")

print(f"Precision: {metrics.box.mp*100:.2f}%")
print(f"Recall: {metrics.box.mr*100:.2f}%")
print(f"mAP50: {metrics.box.map50*100:.2f}%")
print(f"mAP50-95: {metrics.box.map*100:.2f}%")

print("\nDone.")