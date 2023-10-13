import cv2
import torch
import base64
from PIL import Image
import numpy as np
import json
from datetime import datetime
import logging
from flask import Flask, request, redirect

app = Flask(__name__)

model = torch.hub.load(
    "./yolov5", "custom", path="./best.pt", source="local", force_reload=True
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

    if label == 3 or label == 4:  # 라벨 데이터 변경 하드코딩
        labal -= 1

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
def imageHandler():
    now = datetime.now()

    reqJson = request.get_json()

    frameBase64 = reqJson["img"]

    frameData = base64.b64decode(frameBase64)
    frameNp = cv2.imdecode(np.frombuffer(frameData, np.uint8), cv2.IMREAD_COLOR)
    framePil = Image.fromarray(frameNp)

    results = model(framePil)

    imageWidth = frameNp.shape[1]
    imageHeight = frameNp.shape[0]

    maxConf = max([bbox[0][4].tolist() for bbox in zip(results.xyxy[0])])

    for bbox in zip(results.xyxy[0]):
        if bbox[0][4].tolist() == maxConf:
            bboxCoords, frameNp = calcImage(bbox, frameNp, imageWidth, imageHeight, now)

    boundImage = cv2.cvtColor(frameNp, cv2.COLOR_RGB2BGR)
    boundImage = base64.b64encode(cv2.imencode(".jpg", boundImage)[1]).decode("utf-8")

    resultData = json.dumps(bboxCoords, default=str)
    LOGGER.info(resultData)

    return json.dumps({"result": resultData, "img": boundImage})


@app.route("/", methods=["GET"])
def index():
    return redirect("https://github.com/suk-6/AIFrontier")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="10000")
