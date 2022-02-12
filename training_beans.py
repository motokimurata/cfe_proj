# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import tkinter, tkinter.filedialog, tkinter.messagebox
import pickle

## 画像データのクラスIDとパスを取得
#
# @param dir_path 検索ディレクトリ
# @return data_sets [クラスID, 画像データのパス]のリスト
def getDataSet(dir_path):
    data_sets = []    

    sub_dirs = os.listdir(dir_path)
    for classId in sub_dirs:
        sub_dir_path = dir_path + '/' + classId
        img_files = os.listdir(sub_dir_path)
        for f in img_files:
            data_sets.append([classId, sub_dir_path + '/' + f])

    return data_sets


def train_data(data_path):
    """
    main
    """
    # 定数定義
    GRAYSCALE = 0
    # KAZE特徴量抽出器
    detector = cv2.KAZE_create()

    """
    train
    """
    print("train start")
    # 訓練データのパスを取得
    train_set = getDataSet(data_path)
    # 辞書サイズ
    dictionarySize = 2
    # Bag Of Visual Words分類器
    bowTrainer = cv2.BOWKMeansTrainer(dictionarySize)

    # 各画像を分析
    for i, (classId, data_path) in enumerate(train_set):
        # 進捗表示
        sys.stdout.write(".")
        # グレースケールで画像読み込み
        gray = cv2.imread(data_path, GRAYSCALE)
        # 特徴点とその特徴を計算
        keypoints, descriptors= detector.detectAndCompute(gray, None)
        # intからfloat32に変換
        descriptors = descriptors.astype(np.float32)
        # 特徴ベクトルをBag Of Visual Words分類器にセット
        bowTrainer.add(descriptors)

    # Bag Of Visual Words分類器で特徴ベクトルを分類
    codebook = bowTrainer.cluster()
    # 訓練完了
    print("train finish")
    return codebook


if __name__ == '__main__':
    train_data('train_img')
    np.save('bean_traning.npy',codebook)

