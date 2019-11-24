from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import json
from keras.models import model_from_json

if __name__ == '__main__':

    n_epochs = 12

    # データのロード
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(x_train.shape[0], '件のトレーニングデータと ', end='')
    print(y_test.shape[0], '件のテストデータをロードしました')

    # 前処理
    # 画像を1次元化（元の画像は 28 * 28ピクセル： 28x28 = 784）
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # 画像を0-1の範囲に変換（つまり正規化）する
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    print(x_train.shape[0], 'train samples')
    print(y_test.shape[0], 'test samples')

    # 正解ラベルを one-hot-encoding する
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)


    # 保存したモデルの読み込み（10-save-model.py で作成された）
    model = model_from_json(open('number_model.json').read())
    # 保存した重みの読み込み
    model.load_weights('number_predict.hdf5')
    model.summary()

    # モデルを読み込んだら，学習やテストの前にコンパイルが必要
    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    # モデルを評価する（テストデータを使う）
    score = model.evaluate(x_test, y_test)
    print(score)
    print(model.metrics_names)
    print(model.metrics_names[0], " : ", score[0])
    print(model.metrics_names[1], " : ", score[1])
