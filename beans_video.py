import imutils
import numpy as np
import cv2
import tkinter, tkinter.filedialog, tkinter.messagebox
from assets.split_beans import detect_contour
from assets.getpath import getDataSet

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

    # カメラ画像を幅400pxにリサイズ
    #img = imutils.resize(frame, width=400)
    img = frame
    # グレースケール画像へ変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2値化
    retval, bw = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

  # 輪郭を抽出
  #   contours : [領域][Point No][0][x=0, y=1]
  #   cv2.CHAIN_APPROX_NONE: 中間点も保持する
  #   cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # 各輪郭に対する処理
    for i in range(0, len(contours)):

        # 輪郭の領域を計算
        area = cv2.contourArea(contours[i])

        # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
        if area < 1e3 or 1e5 < area:
            continue

        # 外接矩形
        if len(contours[i]) > 0:
            rect = contours[i]
            x, y, w, h = cv2.boundingRect(rect)

            if w < 10 or  h < 10:
                continue
            # 背景差分する範囲を指定
            trim = img[y:(y+h),x:(x+w)]
            # グレースケールで読み込み
            material = cv2.cvtColor(trim, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors= detector.detectAndCompute(material, None)
            #intからfloat32に変換
            try:
                descriptors = descriptors.astype(np.float32)
            except:
                continue
            # Bag Of Visual Wordsの計算
            bowDescriptors = bowExtractor.compute(material, keypoints)
            
            # 結果表示
            className = {"0": "True",
                            "1": "False"}

            actual = "???"    
            if bowDescriptors[0][0] > bowDescriptors[0][1]:
                actual = className["0"]
            elif bowDescriptors[0][0] < bowDescriptors[0][1]:
                actual = className["1"]

            if actual == "True":    
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 10)
                cv2.putText(img, 'True', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 8, 4)
            elif actual == 'False':
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 10)
                cv2.putText(img, 'False', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8, 4)
            else:
                continue

    cv2.imshow("Beans Detection", img)
    k = cv2.waitKey(1)&0xff
    if k == ord('s'):
        cv2.imwrite("./output.jpg", img) # ファイル保存
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()