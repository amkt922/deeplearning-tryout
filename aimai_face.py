#coding: utf-8
import glob
import cv2
print("face start")

cascade_path = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"

for aimai in ['ai', 'mai']:
    for im in glob.glob('./' + aimai + 'data/*.jpg'):
        image = cv2.imread(im)
        #グレースケール変換
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #カスケード分類器の特徴量を取得する
        cascade = cv2.CascadeClassifier(cascade_path)

        #物体認識（顔認識）の実行
        facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.001, minNeighbors=2, minSize=(1, 1))

        print("face rectangle")
        print(facerect )
        if len(facerect) > 0:
            rect = facerect[0]
            #顔だけ切り出して保存
            x = rect[0]
            y = rect[1]
            width = 25 # rect[2]
            height = 25 # rect[3]
            dst = image[y:y+height, x:x+width]
            new_image_path = './' + aimai + 'data/face/' + im.split("/")[len(im.split("/")) - 1]
            cv2.imwrite(new_image_path, dst)
