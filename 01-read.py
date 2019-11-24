from keras.datasets import mnist

def print_image(idx):
    print('-------------------')
    i = 1
    j = 1
    print("   |", end='')
    while j <= len(x_train[idx][0]):
        col = "{:3d}".format(j)
        print(col, end='')
        j += 1
    print("")
    j = 1
    while j<= len(x_train[idx][0])*3+4:
        print('-', end='')
        j += 1
    print("")
    for x_t0 in x_train[idx]:
        row = "{:3d}|".format(i)
        print(row, end='')
        for x_t in x_t0:
            color = "{:3d}".format(x_t)
            print(color, end="")
        print("")
        i += 1

# データのロード
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape[0], '件のトレーニングデータと ', end='')
print(y_test.shape[0], '件のテストデータをロードしました')

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
