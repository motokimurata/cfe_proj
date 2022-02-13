# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tkinter, tkinter.filedialog, tkinter.messagebox
from assets.split_beans import detect_contour
from assets.getpath import getDataSet

"""
photo_get
"""
print("photo get start")


# ファイル選択ダイアログの表示
root = tkinter.Tk()
root.withdraw()

# ここの1行を変更　fTyp = [("","*")] →　fTyp = [("","*.csv")]
fTyp = [("","*.jpg")]
curdir = os.getcwd()
iDir = curdir + '/photo'
file = tkinter.filedialog.askopenfilename(filetypes = fTyp ,initialdir = iDir)

detect_contour(file) # spilit_beans.pyの実行
print("photo get finish")

codebook = np.load('bean_traning.npy')

"""
test
"""
print("test start")
# テストデータのパス取得
test_set = getDataSet("output")


# KNNを使って総当たりでマッチング
matcher = cv2.BFMatcher()
detector = cv2.KAZE_create()
# 定数定義
GRAYSCALE = 0

# Bag Of Visual Words抽出器
bowExtractor = cv2.BOWImgDescriptorExtractor(detector, matcher)
# トレーニング結果をセット
bowExtractor.setVocabulary(codebook)

# 正しく学習できたか検証する
for i, (classId, data_path) in enumerate(test_set):
    # グレースケールで読み込み
    gray = cv2.imread(data_path, GRAYSCALE)
    # 特徴点と特徴ベクトルを計算
    keypoints, descriptors= detector.detectAndCompute(gray, None)
    # intからfloat32に変換
    descriptors = descriptors.astype(np.float32)
    # Bag Of Visual Wordsの計算
    bowDescriptors = bowExtractor.compute(gray, keypoints)
    
    # 結果表示
    className = {"0": "True",
                    "1": "False"}

    actual = "???"    
    if bowDescriptors[0][0] > bowDescriptors[0][1]:
        actual = className["0"]
    elif bowDescriptors[0][0] < bowDescriptors[0][1]:
        actual = className["1"]

    result = ""
    if actual == "???":
        result = " => unknown."
    elif className[classId] == actual:
        result = " => success!!"
    else:
        result = " => fail"

    print("expected: ", className[classId], ", actual: ", actual, result)

