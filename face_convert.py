#coding: utf-8
import glob
import cv2
import pickle 
import numpy as np
from PIL import Image
print("start")

X = []
y = []
for aimai in ['ai', 'mai', 'other']:
    for im in glob.glob('./train/' + aimai + '/*.*'):
        img = Image.open(im)
        X.append(np.array(img).reshape((1, 25*25)))
        if aimai == 'ai':
            y.append(0)
        elif aimai == 'mai':
            y.append(1)
        elif aimai == 'other':
            y.append(2)
        else:
            raise Error("data error") 

X = np.vstack(X)

print("X len {0}".format(len(X)))
print("X.size {0}".format(X.size))
X = X.reshape((len(X),1, 25, 25))
pickle.dump(X, open('X', 'wb'))
pickle.dump(y, open('y', 'wb'))

'''
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
