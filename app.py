import cv2
import torch
import base64
from PIL import Image
import numpy as np
import json
from datetime import datetime
import logging
from flask import Flask, send_file
import io

app = Flask(__name__)

model = torch.hub.load(
    ".", "custom", path="./models/model.pt", source="local", force_reload=True
)

# LOGGER
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger()

# 바운딩 박스 설정
color = (0, 255, 0)
thickness = 2


def calcImage(bbox, frameNp, imageWidth, imageHeight, now):
    xmin, ymin, xmax, ymax, conf, label = bbox[0].tolist()

    bboxCoords = {
        "requestTime": now.strftime("%Y-%m-%d_%H:%M:%S"),
        "conf": conf,
        "label": int(label),
        "imageWidth": imageWidth,
        "imageHeight": imageHeight,
    }

    cv2.rectangle(
        frameNp,
        (int(xmin), int(ymin)),
        (int(xmax), int(ymax)),
        color,
        thickness,
    )

    return bboxCoords, frameNp


@app.route("/api/image", methods=["POST"])
def imageHandler(frameBase64):
    now = datetime.now()

    frameData = base64.b64decode(frameBase64)
    frameNp = cv2.imdecode(np.frombuffer(frameData, np.uint8), cv2.IMREAD_COLOR)
    framePil = Image.fromarray(frameNp)

    results = model(framePil)

    imageWidth = frameNp.shape[1]
    imageHeight = frameNp.shape[0]

    confs = []

    for bbox in zip(results.xyxy[0]):
        confs.append(bbox[0][4].tolist())

    maxConf = max(confs)
    bbox = [bbox for bbox in zip(results.xyxy[0]) if bbox[0][4].tolist() == maxConf]

    bboxCoords, frameNp = calcImage(bbox, frameNp, imageWidth, imageHeight, now)

    jsonData = json.dumps(bboxCoords, default=str)
    boundImage = Image.fromarray(frameNp)
    data = io.BytesIO()
    boundImage.save(data, "JPEG")

    LOGGER.info(jsonData)

    return send_file(
        data,
        mimetype="image/jpeg",
        attachment_filename="result.jpg",
        as_attachment=True,
    )
