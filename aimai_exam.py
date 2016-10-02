#coding: utf-8
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
from sklearn.cross_validation import train_test_split
import pickle
import six
import glob
from PIL import Image, ImageDraw
import cv2

print("exam start")

with open('model.pkl', 'rb') as i:
    model = pickle.load(i)

def predict(x_data, train=False):
    x = chainer.Variable(x_data)
    h = F.relu(model.conv1(x))
    h = F.max_pooling_2d(F.relu(model.conv2(h)), 2)
    h = F.relu(model.conv3(h))
    h = F.max_pooling_2d(F.relu(model.conv4(h)), 2)
    h = F.relu(model.conv5(h))
    h = F.max_pooling_2d(F.relu(model.conv6(h)), 2)
    h = F.dropout(F.relu(model.l1(h)), train=train)
    y = model.l2(h)
    
    y_trimmed = y.data.argmax(axis=1)
    return np.array(y_trimmed, dtype=np.int32)

cascade_path = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"


for index, im in enumerate(glob.glob('./exam/*.JPG')):
    print(im)
    img_color = Image.open(im)
    img = Image.open(im).convert('L')
    try:
        # colorでみる. LにするとExif情報がなくなる
        exif = img_color._getexif()
        orientation = exif.get(0x112, 1)
        if orientation > 1:
            img = img.transpose(Image.ROTATE_270)
            img_color = img_color.transpose(Image.ROTATE_270)
    except AttributeError:
        pass

    #カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)

    #物体認識（顔認識）の実行
    facerect = cascade.detectMultiScale(np.array(img), scaleFactor=1.1, minNeighbors=1, minSize=(10, 10))
    if len(facerect) == 0:
        continue

    print(facerect)
    for rect in facerect:
        data_train = []
        label_train = [] #  dummy

        crop_size = (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
        img2 = img.crop(crop_size)
        img2.thumbnail((25,25), Image.ANTIALIAS)
        try:
            data_train.append(np.asarray(img2).reshape((1, 25*25)).astype(np.float32))
        except ValueError:
            continue
        label_train.append(0)

        data_train_s, data_test_s, label_train_s, label_test_s = train_test_split(data_train, label_train, test_size=0.0)

        x_train = np.array(data_train_s).astype(np.float32)
        y_train = np.array(label_train_s).astype(np.float32)

        x_train /= 255

        # 画像を (nsample, channel, height, width) の4次元テンソルに変換
        x_train = x_train.reshape((len(x_train), 1, 25, 25))
        scores = predict(x_train)

        draw = ImageDraw.Draw(img_color)
        if scores == 0:
            # ai
            draw.rectangle((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]), outline=(0xff, 0x00, 0x00)) 
        elif scores == 1:
            # mai
            draw.rectangle((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]), outline=(0x00, 0xff, 0x00)) 
        else:
            # unknown
            draw.rectangle((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]), outline=(0x00, 0x00, 0xff)) 
    img_color.save("./exam/result/result{0}.png".format(index))
        
print("done")
