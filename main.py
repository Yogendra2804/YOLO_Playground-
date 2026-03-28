# Test this over main.py file and check the res.


from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO # type: ignore

import shutil
import os
import uuid
import cv2 # type: ignore
from collections import Counter

app = FastAPI()

# ----------------------------
# LOAD MODEL
# ----------------------------

model = YOLO("yolo26s.pt")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------------------
# SERVE STATIC HTML
# ----------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/Index.html")


# ----------------------------
# VIDEO UPLOAD + DETECTION
# ----------------------------

@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    # ---- safe filename ----
    ext = os.path.splitext(file.filename)[1]
    safe_name = str(uuid.uuid4()) + ext

    file_path = os.path.join(UPLOAD_FOLDER, safe_name)

    # ---- save uploaded file ----
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ----------------------------
    # YOLO TRACKING
    # ----------------------------

    results = model.track(
        source=file_path,
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        conf=0.6,
        iou=0.5
    )

    final_path = os.path.join(OUTPUT_FOLDER, safe_name)

    allowed_classes = ["cat", "dog", "cow", "person"]

    counted_ids = set()
    track_votes = {}

    counts = {
        "cat": 0,
        "dog": 0,
        "cow": 0,
        "person": 0
    }

    # ----------------------------
    # VIDEO SETUP
    # ----------------------------

    cap = cv2.VideoCapture(file_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(final_path, fourcc, fps, (width, height))

    # ----------------------------
    # PROCESS FRAMES
    # ----------------------------

    for r in results:

        if r.boxes is not None:

            for box in r.boxes:

                if box.id is None:
                    continue

                conf = float(box.conf)

                # ignore weak detections
                if conf < 0.55:
                    continue

                cls = int(box.cls)
                name = model.names[cls]

                if name not in allowed_classes:
                    continue

                track_id = int(box.id)

                # ----------------------------
                # TEMPORAL VOTING
                # ----------------------------

                if track_id not in track_votes:
                    track_votes[track_id] = []

                track_votes[track_id].append(name)

                # keep last 10 predictions
                track_votes[track_id] = track_votes[track_id][-10:]

                # most frequent class
                name = Counter(track_votes[track_id]).most_common(1)[0][0]

                # ----------------------------
                # COUNT UNIQUE OBJECTS
                # ----------------------------

                if track_id not in counted_ids:

                    counted_ids.add(track_id)
                    counts[name] += 1

        frame = r.plot()

        out.write(frame)

    cap.release()
    out.release()

    # ----------------------------
    # RETURN RESULT
    # ----------------------------

    return JSONResponse({
        "animal": list(counts.keys()),
        "count": counts,
        "file": safe_name
    })


# ----------------------------
# DOWNLOAD OUTPUT VIDEO
# ----------------------------

@app.get("/output/{filename}")
def get_output(filename: str):

    return FileResponse(
        os.path.join(OUTPUT_FOLDER, filename),
        media_type="video/mp4"
    )