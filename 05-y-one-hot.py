from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

# 平べったい配列であるが，表示では改行を入れる
def print_image(idx):
    print('-------------------')
    i = 1
    for x_t in x_train[idx]:
        dat = "{0:.2f} ".format(x_t)
        if dat == '0.00 ':
            dat = '     '
        print(dat, end='')
        if (i % 28 == 0):
            print("")
        i += 1

    print("")
    print("-------------------")
    print("この文字の正解ラベル：", y_train[idx])

if __name__ == '__main__':

    # データのロード
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(x_train.shape[0], '件のトレーニングデータと ', end='')
    print(y_test.shape[0], '件のテストデータをロードしました')

    # 前処理
    # 画像を1次元化
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


    ## データを取得する
    while True:
        print('-------------------')
        print('表示したいイメージの番号（0 から', x_train.shape[0]- 1 , 'まで）を入力してください（-1で終了します）: ', end="")
        str_idx = input()

        if str_idx == "":
            print('入力してください')
            continue
        # 入力した文字を整数に変換するが，ここでは例外処理が必要
        try:
            idx = int(str_idx)
        except ValueError:
            print("エラー：数字以外の文字は入力できません")
            continue
        if idx == -1:
            # print("終了します")
            break
        if idx < 0 or idx > x_train.shape[0]- 1:
            print('正しい値を入れてください')
            continue
        print_image(idx)

    print("-----終了しました------")
