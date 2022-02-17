import cv2
import numpy as np
from PIL import Image

# 指定した画像(path)の物体を検出し、外接矩形の画像を出力します
def detect_contour(path):

  # 画像を読込
  src = cv2.imread(path, cv2.IMREAD_COLOR)

  # グレースケール画像へ変換
  gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

  # 2値化
  retval, bw = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

  # 輪郭を抽出
  #   contours : [領域][Point No][0][x=0, y=1]
  #   cv2.CHAIN_APPROX_NONE: 中間点も保持する
  #   cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない
  contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

  # 矩形検出された数（デフォルトで0を指定）
  detect_count = 0

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
      cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

      # 外接矩形毎に画像を保存
      cv2.imwrite('output/0/photo' + str(detect_count) + '.jpg', src[y:y + h, x:x + w])
      img = Image.open('output/0/photo' + str(detect_count) + '.jpg') #画像サイズを256×256へ変更
      img_resize = img.resize((256, 256))
      img_resize.save('output/0/photo' + str(detect_count) + '.jpg')



      detect_count = detect_count + 1
  
  cv2.waitKey(1)
  # 終了処理
  cv2.destroyAllWindows()
  cv2.waitKey(1)
  
if __name__ == '__main__':
  detect_contour('photo/cfe07.jpg')