#coding: utf-8
import glob
import cv2
import pickle 
import numpy as np
from PIL import Image
print("start")
for aimai in ['mai']:
    path = './train/' + aimai
    for im in glob.glob(path + '/gray_*.jpg'):
        file_name = im.split("/")[len(im.split("/")) - 1]
        img = Image.open(im)
        #上下反転
        tmp = img.transpose(Image.FLIP_TOP_BOTTOM)
        tmp.save(path + "/flip_" + file_name)
        #90度回転
        tmp = img.transpose(Image.ROTATE_90)
        tmp.save(path + "/spin90_" + file_name)
        #270度回転
        tmp = img.transpose(Image.ROTATE_270)
        tmp.save(path + "/spin270_" + file_name)
        #左右反転
        tmp = img.transpose(Image.FLIP_LEFT_RIGHT)
        tmp.save(path + "/flipLR_" + file_name)
'''

cascade_path = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
for aimai in ['face_detect']:
    for i, im in enumerate(glob.glob('./' + aimai + '/*.JPG')):
        file_name = im.split("/")[len(im.split("/")) - 1]
        img = Image.open(im).convert('L')
        img.save(im)
        #上下反転
        tmp = img.transpose(Image.FLIP_TOP_BOTTOM)
        tmp.save("./face_detect/flip_" + file_name)

        print(im)
        # image = cv2.imread(Image.open(im).resize((480,360)))
        image = np.array(Image.open(im).resize((480,360)))
        #グレースケール変換
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #カスケード分類器の特徴量を取得する
        cascade = cv2.CascadeClassifier(cascade_path)

        #物体認識（顔認識）の実行
        facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(1, 1))
        for j, rect in enumerate(facerect):
            print(facerect )
            #顔だけ切り出して保存
            x = rect[0]
            y = rect[1]
            width = 25 # rect[2]
            height = 25 # rect[3]
            dst = image[y:y+height, x:x+width]
            new_image_path = './' + aimai + '/face_alt_' + str(i) + "_"  + str(j) + '_' + im.split("/")[len(im.split("/")) - 1]
            cv2.imwrite(new_image_path, dst)

'''
'''
X = np.vstack(X)

print(X.size)
print(len(X))
print(len(X) * 1 * 25 * 25)
X = X.reshape((len(X),1, 25, 25))
pickle.dump(X, open('X', 'wb'))
pickle.dump(y, open('y', 'wb'))

cascade_path = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
for i, im in enumerate(glob.glob('./jaffe/*.tiff')):
    img = Image.open(im).convert('L')
    img.thumbnail((25,25), Image.ANTIALIAS)
    img.save('./otherdata/jeff' + str(i) + '.jpg')

for aimai in ['ai', 'mai']:
    for im in glob.glob('./' + aimai + 'data/face/*.jpg'):
        img = Image.open(im).convert('L')
        new_image_path = './' + aimai + 'data/face/gray_' + im.split("/")[len(im.split("/")) - 1]
        img.save(new_image_path)
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
'''
print("end")
