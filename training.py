#coding: UTF-8
import pandas as pd
import numpy as np
import chainer
import math
import time
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
from chainer import Chain, Variable
from chainer import optimizers
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from NeuralNet import NeuralNet


'''
学習,x:data=[[],[]],t:label=[]
'''
def training(x, t, test_rate=0, batch_size=None, n_epoch=100):
    # numpy配列変換
    x = np.array(x).astype(np.float32)
    t = np.array(t).astype(np.int32)

    x_test = x[:int(test_rate*x.shape[0])]
    x_train = x[int(test_rate*x.shape[0]):]
    t_test = t[:int(test_rate*t.shape[0])]
    t_train = t[int(test_rate*t.shape[0]):]
    print(x_train)
    print(x_test)
    N = x_train.shape[0]
    # バッチサイズデフォルト設定
    if batch_size == None:
        batch_size = N
    # モデルのセットアップ
    model = NeuralNet()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # loss/accuracy格納配列
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    # 時間を測定
    start_time = time.time()

    # 学習ループ
    for epoch in range(1, n_epoch + 1):
        perm = np.random.permutation(N)
        sum_loss = 0
        sum_accuracy = []
        for i in range(0, N, batch_size):
            x_batch = x_train[perm[i:i + batch_size]]
            t_batch = t_train[perm[i:i + batch_size]]

            model.cleargrads()
            loss = model.loss(x_batch, t_batch)
            accuracy = model.accuracy(t_batch)
            loss.backward()
            optimizer.update()
            sum_loss += loss.data * batch_size
            sum_accuracy.append(accuracy.data)

        # 学習誤差/精度の平均を計算
        ave_loss = sum_loss / N
        train_loss.append(ave_loss)
        train_accuracy.append(sum(sum_accuracy) / len(sum_accuracy))

        # テスト誤差
        if len(x_test) != 0:
            loss = model.loss(x_test, t_test)
            accuracy = model.accuracy(t_test)
            test_loss.append(loss.data)
            test_accuracy.append(accuracy.data)

        # 学習過程を出力
        if epoch % 100 == 1:
            print("Ep/MaxEp     train_loss     test_loss")

        if epoch % 10 == 0:
            print("{:4}/{}  {:10.5}   {:10.5}".format(epoch,
                                                      n_epoch, ave_loss, float(loss.data)))

            # 誤差をリアルタイムにグラフ表示
            plt.plot(train_loss, label="training")
            plt.plot(test_loss, label="test")
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.pause(0.1)
            plt.clf()

    # 経過時間
    total_time = int(time.time() - start_time)
    print("Time : {} [s]".format(total_time))

    # 誤差のグラフ作成
    plt.figure(figsize=(4, 3))
    plt.plot(train_loss, label="training")
    plt.plot(test_loss, label="test")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("result/loss_history.png")
    plt.clf()
    plt.close()

    # 精度のグラフ作成
    plt.figure(figsize=(4, 3))
    plt.plot(train_accuracy, label="training")
    plt.plot(test_accuracy, label="test")
    plt.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig("result/accuracy_history.png")
    plt.clf()
    plt.close()

    # 学習済みモデルの保存
    serializers.save_npz("model/model.npz", model)

# 特徴量x=[[],[]],正解値t=[]
def make_dataset(data_path="./data/sample.csv"):
    df_data_path = pd.read_csv(data_path, header=None)
    x = df_data_path.iloc[:, 0:-1]
    x = x.values.tolist()
    t = df_data_path.iloc[:, -1]
    t = t.values.tolist()
    return x, t


if __name__ == "__main__":
    x, t = make_dataset(data_path="./data/sample.csv")
    training(x, t, test_rate=0, batch_size=10, n_epoch=100)
