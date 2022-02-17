import cv2
import numpy as np
from PIL import Image
import os
import shutil
import tkinter, tkinter.filedialog, tkinter.messagebox


codebook = np.load('bean_traning.npy')
matcher = cv2.BFMatcher()
detector = cv2.KAZE_create()
# 定数定義
GRAYSCALE = 0

# Bag Of Visual Words抽出器
bowExtractor = cv2.BOWImgDescriptorExtractor(detector, matcher)
# トレーニング結果をセット
bowExtractor.setVocabulary(codebook)

# 画像を読込
src = cv2.imread('photo/cfe07.jpg', cv2.IMREAD_COLOR)

# グレースケール画像へ変換
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# 2値化
retval, bw = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 輪郭を抽出
#   contours : [領域][Point No][0][x=0, y=1]
#   cv2.CHAIN_APPROX_NONE: 中間点も保持する
#   cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない
contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# 各輪郭に対する処

# 外接矩形
rect = contours[10]
x, y, w, h = cv2.boundingRect(rect)
print(x)
print(y+h)
print(w)
print(h)

# 背景差分する範囲を指定
a = src[y:y + h, x:x + w]
# グレースケールで読み込み
cv2.imshow('gray',a)
cv2.waitKey(0) #ここで初めてウィンドウが表示される