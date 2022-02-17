import imutils
import numpy as np
import cv2
import tkinter, tkinter.filedialog, tkinter.messagebox
from assets.detect_beans import detect_cam

# VideoCaptureをオープン
cap = cv2.VideoCapture('rtsp://192.168.11.12/')
codebook = np.load('bean_traning.npy')
matcher = cv2.BFMatcher()
detector = cv2.KAZE_create()
# Bag Of Visual Words抽出器
bowExtractor = cv2.BOWImgDescriptorExtractor(detector, matcher)
# トレーニング結果をセット
bowExtractor.setVocabulary(codebook)

# カメラ画像を読み込み，顔検出して表示するループ
while True:
    ret, frame = cap.read()
    
    img = detect_cam(frame,detector,bowExtractor) # 画像認識にかける

    cv2.imshow("Beans Detection", img)
    k = cv2.waitKey(1)&0xff
    if k == ord('s'):
        cv2.imwrite("./output.jpg", img) # ファイル保存
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()