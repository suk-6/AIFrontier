import os
import shutil
from PIL import Image
from tqdm import tqdm
import random
import math
import cv2

rootDir = os.getcwd()
originDir = "/Users/woosuk/Downloads/origin/obj_train_data"
print(rootDir)


def moveFiles():
    print("moveFiles")

    if not os.path.exists(f"{rootDir}/images"):
        os.mkdir(f"{rootDir}/images")
    if not os.path.exists(f"{rootDir}/labels"):
        os.mkdir(f"{rootDir}/labels")

    for root, _, files in os.walk(originDir):
        for fname in tqdm(files):
            if (
                fname.endswith(".jpg")
                or fname.endswith(".jpeg")
                or fname.endswith(".png")
            ):
                img = Image.open(f"{root}/{fname}").convert("RGB")
                img.save(f"{rootDir}/images/{fname.split('.')[0]}.jpg")
            elif fname.endswith(".txt"):
                shutil.copy(f"{root}/{fname}", f"{rootDir}/labels/{fname}")


def verifyDataset():
    print("verifyDataset")
    for _, _, files in os.walk(f"{rootDir}/images"):
        for fname in tqdm(files):
            if not os.path.exists(f"{rootDir}/labels/{fname.split('.')[0]}.txt"):
                print(f"labels/{fname.split('.')[0]}.txt")

    for _, _, files in os.walk(f"{rootDir}/labels"):
        for fname in tqdm(files):
            if not os.path.exists(f"{rootDir}/images/{fname.split('.')[0]}.jpg"):
                print(f"images/{fname.split('.')[0]}.jpg")


def splitDataset():
    print("splitDataset")

    if not os.path.exists(f"{rootDir}/dataset"):
        os.mkdir(f"{rootDir}/dataset")

    if not os.path.exists(f"{rootDir}/dataset/train"):
        os.mkdir(f"{rootDir}/dataset/train")
    if not os.path.exists(f"{rootDir}/dataset/train/images"):
        os.mkdir(f"{rootDir}/dataset/train/images")
    if not os.path.exists(f"{rootDir}/dataset/train/labels"):
        os.mkdir(f"{rootDir}/dataset/train/labels")

    if not os.path.exists(f"{rootDir}/dataset/val"):
        os.mkdir(f"{rootDir}/dataset/val")
    if not os.path.exists(f"{rootDir}/dataset/val/images"):
        os.mkdir(f"{rootDir}/dataset/val/images")
    if not os.path.exists(f"{rootDir}/dataset/val/labels"):
        os.mkdir(f"{rootDir}/dataset/val/labels")

    if not os.path.exists(f"{rootDir}/dataset/test"):
        os.mkdir(f"{rootDir}/dataset/test")
    if not os.path.exists(f"{rootDir}/dataset/test/images"):
        os.mkdir(f"{rootDir}/dataset/test/images")
    if not os.path.exists(f"{rootDir}/dataset/test/labels"):
        os.mkdir(f"{rootDir}/dataset/test/labels")

    targetList = []
    for _, _, files in os.walk(f"{rootDir}/images"):
        for fname in tqdm(files):
            targetList.append(fname.split(".")[0])

    random.shuffle(targetList)

    trainList = targetList[: math.floor(len(targetList) * 0.8)]
    valList = targetList[
        math.floor(len(targetList) * 0.8) : math.floor(len(targetList) * 0.9)
    ]
    testList = targetList[math.floor(len(targetList) * 0.9) :]

    for fname in tqdm(trainList):
        shutil.copy(
            f"{rootDir}/images/{fname}.jpg",
            f"{rootDir}/dataset/train/images/{fname}.jpg",
        )
        shutil.copy(
            f"{rootDir}/labels/{fname}.txt",
            f"{rootDir}/dataset/train/labels/{fname}.txt",
        )

    for fname in tqdm(valList):
        shutil.copy(
            f"{rootDir}/images/{fname}.jpg",
            f"{rootDir}/dataset/val/images/{fname}.jpg",
        )
        shutil.copy(
            f"{rootDir}/labels/{fname}.txt",
            f"{rootDir}/dataset/val/labels/{fname}.txt",
        )

    for fname in tqdm(testList):
        shutil.copy(
            f"{rootDir}/images/{fname}.jpg",
            f"{rootDir}/dataset/test/images/{fname}.jpg",
        )
        shutil.copy(
            f"{rootDir}/labels/{fname}.txt",
            f"{rootDir}/dataset/test/labels/{fname}.txt",
        )


def verifySplitDataset():
    print("verifySplitDataset")
    for _, _, files in os.walk(f"{rootDir}/dataset/train/images"):
        for fname in tqdm(files):
            if not os.path.exists(
                f"{rootDir}/dataset/train/labels/{fname.split('.')[0]}.txt"
            ):
                print(f"train/labels/{fname.split('.')[0]}.txt")

    for _, _, files in os.walk(f"{rootDir}/dataset/train/labels"):
        for fname in tqdm(files):
            if not os.path.exists(
                f"{rootDir}/dataset/train/images/{fname.split('.')[0]}.jpg"
            ):
                print(f"train/images/{fname.split('.')[0]}.jpg")

    for _, _, files in os.walk(f"{rootDir}/dataset/val/images"):
        for fname in tqdm(files):
            if not os.path.exists(
                f"{rootDir}/dataset/val/labels/{fname.split('.')[0]}.txt"
            ):
                print(f"val/labels/{fname.split('.')[0]}.txt")

    for _, _, files in os.walk(f"{rootDir}/dataset/val/labels"):
        for fname in tqdm(files):
            if not os.path.exists(
                f"{rootDir}/dataset/val/images/{fname.split('.')[0]}.jpg"
            ):
                print(f"val/images/{fname.split('.')[0]}.jpg")

    for _, _, files in os.walk(f"{rootDir}/dataset/test/images"):
        for fname in tqdm(files):
            if not os.path.exists(
                f"{rootDir}/dataset/test/labels/{fname.split('.')[0]}.txt"
            ):
                print(f"test/labels/{fname.split('.')[0]}.txt")

    for _, _, files in os.walk(f"{rootDir}/dataset/test/labels"):
        for fname in tqdm(files):
            if not os.path.exists(
                f"{rootDir}/dataset/test/images/{fname.split('.')[0]}.jpg"
            ):
                print(f"test/images/{fname.split('.')[0]}.jpg")


def drawBoundingBox():
    print("drawBoundingBox")

    if not os.path.exists(f"{rootDir}/verifyImage"):
        os.mkdir(f"{rootDir}/verifyImage")

    for _, _, files in os.walk(f"{rootDir}/images"):
        for fname in tqdm(files):
            if fname.endswith(".jpg"):
                img = cv2.imread(f"{rootDir}/images/{fname}")
                f = open(f"{rootDir}/labels/{fname.split('.')[0]}.txt", "r")
                lines = f.readlines()
                for line in lines:
                    line = line.split(" ")
                    x = float(line[1])
                    y = float(line[2])
                    w = float(line[3])
                    h = float(line[4])
                    x1 = int((x - w / 2) * img.shape[1])
                    y1 = int((y - h / 2) * img.shape[0])
                    x2 = int((x + w / 2) * img.shape[1])
                    y2 = int((y + h / 2) * img.shape[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img.save(f"{rootDir}/verifyImage/{fname}")


def makeZip():
    print("makeZip")

    shutil.copy(f"{rootDir}/data.yaml", f"{rootDir}/dataset/data.yaml")

    if not os.path.exists(f"{rootDir}/zip"):
        os.mkdir(f"{rootDir}/zip")

    shutil.make_archive(f"{rootDir}/zip/dataset", "zip", f"{rootDir}/dataset")
    shutil.make_archive(f"{rootDir}/zip/verifyImage", "zip", f"{rootDir}/verifyImage")


moveFiles()
verifyDataset()

splitDataset()
verifySplitDataset()

drawBoundingBox()

makeZip()
