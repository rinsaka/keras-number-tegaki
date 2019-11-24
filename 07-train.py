from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

if __name__ == '__main__':

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

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=784))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    model.summary()

    # 学習してみよう（このコードだけで，学習状況も表示される）
    model.fit(x_train, y_train,
            batch_size=100,
            epochs=12,
            verbose=1)
