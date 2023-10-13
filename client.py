import requests
import base64
import cv2

image_name = "test.jpg"
img = cv2.imread(image_name)
jpg = cv2.imencode(".jpg", img)
b64String = base64.b64encode(jpg[1]).decode("utf-8")

jsonData = {
    "img": b64String,
}

r = requests.post("http://localhost:10000/api/image", json=jsonData)

print(r.json()["result"])
