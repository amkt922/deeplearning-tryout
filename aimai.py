#coding: utf-8
import argparse
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import time
import corpus
from sklearn.cross_validation import train_test_split
import pickle
import six
print("start")

xp = np

# 学習のパラメータ
batchsize = 100
n_epoch = 40
n_units = 512

# ontents = corpus.get_contents()

# data_train = pickle.load(open('X.pkl', 'rb'))
# label_train = pickle.load(open('y.pkl', 'rb'))
data_train = pickle.load(open('X', 'rb'))
label_train = pickle.load(open('y', 'rb'))


print("got train label data")
print("data len " + str(len(data_train)))

# 学習データと試験データに分けてみる
data_train_s, data_test_s, label_train_s, label_test_s = train_test_split(data_train, label_train, test_size=0.1)

print("train label data are split")

x_train = np.array(data_train_s).astype(np.float32)
y_train = np.array(label_train_s).astype(np.int32)
x_test = np.array(data_test_s).astype(np.float32)
y_test = np.array(label_test_s).astype(np.int32)

x_train /= 255
x_test /= 255

N = len(y_train)
N_test = len(y_test)
# x_train, x_test = np.split(data_train, [N])
# y_train, y_test = np.split(label_train, [N])
# N_test = y_test.size

# 画像を (nsample, channel, height, width) の4次元テンソルに変換
# x_train = x_train.reshape((len(x_train), 3, 71, 40))
# x_test = x_test.reshape((len(x_test), 3, 71, 40))
# 多層パーセプトロンのモデル（パラメータ集合）
'''
model = chainer.FunctionSet(l1=F.Linear(8520, n_units),      # 入力層-隠れ層1
                            l2=F.Linear(n_units, n_units),  # 隠れ層1-隠れ層2
                            l3=F.Linear(n_units, 3))       # 隠れ層2-出力層
                            # l3=F.Linear(n_units, 9))       # 隠れ層2-出力層
'''
if True:
    model = chainer.FunctionSet(
                                conv1=F.Convolution2D(1, 32, 3, pad=1),
                                conv2=F.Convolution2D(32, 32, 3, pad=1),
                                conv3=F.Convolution2D(32, 32, 3, pad=1),
                                conv4=F.Convolution2D(32, 32, 3, pad=1),
                                conv5=F.Convolution2D(32, 32, 3, pad=1),
                                conv6=F.Convolution2D(32, 32, 3, pad=1),
                                l1=F.Linear(512, n_units),        
                                l2=F.Linear(n_units, 3))
else:
    with open('model.pkl', 'rb') as i:
        model = pickle.load(i)

def forward(x_data, y_data, train=True):
    # 順伝播の処理を定義
    # 入力と教師データ
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h = F.relu(model.conv1(x))
    h = F.max_pooling_2d(F.relu(model.conv2(h)), 2)
    h = F.relu(model.conv3(h))
    h = F.max_pooling_2d(F.relu(model.conv4(h)), 2)
    h = F.relu(model.conv5(h))
    h = F.max_pooling_2d(F.relu(model.conv6(h)), 2)
    h = F.dropout(F.relu(model.l1(h)), train=train)
    y = model.l2(h)

    # 訓練時とテスト時で返す値を変える
    if train:
        # 訓練時は損失を返す
        # 多値分類なのでクロスエントロピーを使う
        loss = F.softmax_cross_entropy(y, t)
        return loss
    else:
        # テスト時は精度を返す
        acc = F.accuracy(y, t)
        return acc

# Optimizerをセット
# 最適化対象であるパラメータ集合のmodelを渡しておく
optimizer = optimizers.Adam()
optimizer.setup(model)

fp = open('accuracy.txt', "w")
fp.write("epoch\ttest_accuracy\n")

# 訓練ループ
# 各エポックでテスト精度を求める
print("training start...")
start_time = time.clock()
for epoch in six.moves.range(1, n_epoch + 1):

    print('epoch test {0}'.format(epoch))

    # 訓練データを用いてパラメータを更新する
    perm = np.random.permutation(N)
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        # x_batch = xp.asarray(x_train[perm[i:i + batchsize]])
        x_batch = x_train[perm[i:i + batchsize]]
        # y_batch = xp.asarray(y_train[perm[i:i + batchsize]])
        y_batch = y_train[perm[i:i + batchsize]]

        optimizer.zero_grads()
        loss = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(y_batch)

    print("train mean loss: {0}".format((sum_loss / N)))

    # テストデータを用いて精度を評価する
    sum_accuracy = 0
    for i in six.moves.range(0, N_test, batchsize):
        x_batch = x_test[i:i + batchsize]
        y_batch = y_test[i:i + batchsize]

        acc = forward(x_batch, y_batch, train=False)
        sum_accuracy += float(acc.data) * len(y_batch)

    print ("test accuracy: {0}".format((sum_accuracy / N_test)))
    fp.write("%d\t%f\n" % (epoch, sum_accuracy / N_test))
    fp.flush()

end_time = time.clock()
print (end_time - start_time)

fp.close()

with open('model.pkl', 'wb') as o:
    pickle.dump(model, o)
print("model was saved")
print("done")

