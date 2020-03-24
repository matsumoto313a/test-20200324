
# 10章 深層学習による数字認識モデル


```python
# 必要ライブラリの宣言
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
```


```python
# PDF出力用
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')
```

## 10.7 実装その1

### データ読み込み


```python
# データ読み込み
import os
import urllib.request

mnist_file = 'mnist-original.mat'
mnist_path = 'mldata'
mnist_url = 'https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat'

# ファイルの存在確認
mnist_fullpath = os.path.join('.', mnist_path, mnist_file)
if not os.path.isfile(mnist_fullpath):
    # データダウンロード
    mldir = os.path.join('.', 'mldata')
    os.makedirs(mldir, exist_ok=True)
    print("donwload %s started." % mnist_file)
    urllib.request.urlretrieve(mnist_url, mnist_fullpath)
    print("donwload %s finished." % mnist_file)

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='.')
```


```python
x_org, y_org = mnist.data, mnist.target
```

### 入力データ加工


```python
# 入力データの加工

# step1 データ正規化 値の範囲を[0, 1]とする
x_norm = x_org / 255.0

# 先頭にダミー変数(1)を追加
x_all = np.insert(x_norm, 0, 1, axis=1)

print('ダミー変数追加後', x_all.shape)
```


```python
# step 2 yをOne-hot-Vectorに

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_all_one = ohe.fit_transform(np.c_[y_org])
print('One Hot Vector化後', y_all_one.shape)
```


```python
# step 3 学習データ、検証データに分割

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, y_train_one, y_test_one = train_test_split(
    x_all, y_org, y_all_one, train_size=60000, test_size=10000, shuffle=False)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, 
    y_train_one.shape, y_test_one.shape)
```


```python
# データ内容の確認

N = 20
np.random.seed(123)
indexes = np.random.choice(y_test.shape[0], N, replace=False)
x_selected = x_test[indexes,1:]
y_selected = y_test[indexes]
plt.figure(figsize=(10, 3))
for i in range(N):
    ax = plt.subplot(2, N/2, i + 1)
    plt.imshow(x_selected[i].reshape(28, 28),cmap='gray_r')
    ax.set_title('%d' %y_selected[i], fontsize=16)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

### 予測関数


```python
# シグモイド関数
def sigmoid(x):
    return 1/(1+ np.exp(-x))
```


```python
# softmax関数
def softmax(x):
    x = x.T
    x_max = x.max(axis=0)
    x = x - x_max
    w = np.exp(x)
    return (w / w.sum(axis=0)).T
```

### 評価関数


```python
# 交差エントロピー関数
def cross_entropy(yt, yp):
    return -np.mean(np.sum(yt * np.log(yp), axis=1))
```


```python
# 評価処理 (戻り値は精度と損失関数)
from sklearn.metrics import accuracy_score

def evaluate(x_test, y_test, y_test_one, V, W):
    b1_test = np.insert(sigmoid(x_test @ V), 0, 1, axis=1)
    yp_test_one = softmax(b1_test @ W)
    yp_test = np.argmax(yp_test_one, axis=1)
    loss = cross_entropy(y_test_one, yp_test_one)
    score = accuracy_score(y_test, yp_test)
    return score, loss   
```

### ミニバッチ処理


```python
# ミニバッチ用index取得関数
import numpy as np

class Indexes():
    
    # コンストラクタ
    def __init__(self, total, size):
        # 配列全体の大きさ
        self.total   = total
        # batchサイズ
        self.size    = size
        #　作業用indexes 初期値はNULLにしておく
        self.indexes = np.zeros(0) 

    # index取得関数    
    def next_index(self):
        next_flag = False
        
    # bacthサイズより作業用Indexesが小さい場合はindexes再生成
        if len(self.indexes) < self.size: 
            self.indexes = np.random.choice(self.total, 
                self.total, replace=False)
            next_flag = True
            
        # 戻り用index取得と作業用indexes更新
        index = self.indexes[:self.size]
        self.indexes = self.indexes[self.size:]
        return index, next_flag
```


```python
# Indexesクラスのテスト

# クラス初期化
# 20: 全体の配列の大きさ
# 5: 一回に取得するindexの数
indexes = Indexes(20, 5)

for i in range(6):
    # next_index関数呼び出し
    # 戻り値1:  indexのnumpy配列
    # 戻り値2: 作業用Indexの更新があったかどうか
    arr, flag = indexes.next_index()
    print(arr, flag)
```

### 初期化処理 その1


```python
# 変数初期宣言 初期バージョン

# 隠れ層のノード数
H = 128
H1 = H + 1
# M: 訓練用系列データ総数
M  = x_train.shape[0]
# D: 入力データ次元数
D = x_train.shape[1]
# N: 分類クラス数
N = y_train_one.shape[1]

# 繰り返し回数
nb_epoch = 100
# ミニバッチサイズ
batch_size = 512
B = batch_size
# 学習率
alpha = 0.01

# 重み行列の初期設定(すべて1)
V = np.ones((D, H))
W = np.ones((H1, N))

# 評価結果記録用 (損失関数値と精度)
history1 = np.zeros((0, 3))

# ミニバッチ用関数初期化
indexes = Indexes(M, batch_size)

# 繰り返し回数カウンタ初期化
epoch = 0
```

### メイン処理


```python
# メイン処理
while epoch < nb_epoch:

    # 学習対象の選択(ミニバッチ学習法)
    index, next_flag = indexes.next_index()
    x, yt = x_train[index], y_train_one[index]
    
    # 予測値計算 (順伝播) 
    a = x @ V                         # (10.6.3)
    b = sigmoid(a)                    # (10.6.4)
    b1 = np.insert(b, 0, 1, axis=1)   # ダミー変数の追加 
    u = b1 @ W                        # (10.6.5)   
    yp = softmax(u)                   # (10.6.6)
    
    # 誤差計算 
    yd = yp - yt                      # (10.6.7)
    bd = b * (1-b) * (yd @ W[1:].T)   # (10.6.8)

    # 勾配計算
    W = W - alpha * (b1.T @ yd) / B   # (10.6.9)
    V = V - alpha * (x.T @ bd) / B    # (10.6.10)
    
    # ログ記録用
    if next_flag: # 1 epoch 終了後の処理
        score, loss = evaluate(
            x_test, y_test, y_test_one, V, W)
        history1 = np.vstack((history1, 
            np.array([epoch, loss, score])))
        print("epoch = %d loss = %f score = %f" 
            % (epoch, loss, score))
        epoch = epoch + 1
```

### 結果確認　その1


```python
#損失関数値と精度の確認
print('初期状態: 損失関数:%f 精度:%f' 
        % (history1[0,1], history1[0,2]))
print('最終状態: 損失関数:%f 精度:%f' 
        % (history1[-1,1], history1[-1,2]))
```


```python
# 学習曲線の表示 (損失関数値)
plt.plot(history1[:,0], history1[:,1])
plt.ylim(0,2.5)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()
```


```python
# 学習曲線の表示 (精度)
plt.plot(history1[:,0], history1[:,2])
plt.ylim(0,1)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()
```

## 10.8 パラメータ初期値の設定変更

### 変数初期化　その2


```python
# 変数初期宣言 重み行列の初期化方式変更

# 隠れ層のノード数
H = 128
H1 = H + 1

# M: 訓練用系列データ総数
M  = x_train.shape[0]

# D: 入力データ次元数
D = x_train.shape[1]

# N: 分類クラス数
N = y_train_one.shape[1]

# 機械学習パラメータ
alpha = 0.01
nb_epoch = 100
batch_size = 512
B = batch_size

# 重み行列の初期設定(すべて1)
V = np.ones((D, H))
W = np.ones((H1, N))

# 評価結果記録用 (損失関数値と精度)
history2 = np.zeros((0, 3))

# ミニバッチ用関数初期化
indexes = Indexes(M, batch_size)

# 繰り返し回数カウンタ初期化
epoch = 0
```


```python
# 重み行列の初期設定改訂版
V = np.random.randn(D, H) / np.sqrt(D / 2)
W = np.random.randn(H1, N) / np.sqrt(H1 / 2)
print(V[:2,:5])
print(W[:2,:5])
```

### メイン処理


```python
# メイン処理
while epoch < nb_epoch:
    
    # 学習対象の選択(ミニバッチ学習法)
    index, next_flag = indexes.next_index()
    x, yt = x_train[index], y_train_one[index]

    # 予測値計算 (順伝播) 
    a = x @ V                         # (10.6.3)
    b = sigmoid(a)                    # (10.6.4)
    b1 = np.insert(b, 0, 1, axis=1)   # ダミー変数の追加 
    u = b1 @ W                        # (10.6.5)   
    yp = softmax(u)                   # (10.6.6)
    
    # 誤差計算 
    yd = yp - yt                      # (10.6.7)
    bd = b * (1-b) * (yd @ W[1:].T)   # (10.6.8)

    # 勾配計算
    W = W - alpha * (b1.T @ yd) / B   # (10.6.9)
    V = V - alpha * (x.T @ bd) / B    # (10.6.10)

    if next_flag: # 1epoch 終了後の処理
        score, loss = evaluate(
            x_test, y_test, y_test_one, V, W)
        history2 = np.vstack((history2, 
            np.array([epoch, loss, score])))
        print("epoch = %d loss = %f score = %f" 
            % (epoch, loss, score))
        epoch = epoch + 1
```

### 結果確認　その2


```python
#損失関数値と精度の確認
print('初期状態: 損失関数:%f 精度:%f' 
        % (history2[0,1], history2[0,2]))
print('最終状態: 損失関数:%f 精度:%f' 
        % (history2[-1,1], history2[-1,2]))
```


```python
# 学習曲線の表示 (損失関数値)
plt.plot(history2[:,0], history2[:,1])
plt.ylim(0,2.5)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()
```


```python
# 学習曲線の表示 (精度)
plt.plot(history2[:,0], history2[:,2])
plt.ylim(0,1)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()
```

## 10.9 ReLU関数の導入


```python
# ReLU関数
def ReLU(x):
    return np.maximum(0, x)
```


```python
# step関数
def step(x):
    return 1.0 * ( x > 0)
```


```python
# ReLU関数とstep関数のグラフ表示

xx =  np.linspace(-4, 4, 501)
yy = ReLU(xx)
plt.figure(figsize=(6,6))
#plt.ylim(0.0, 1.0)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)
plt.grid(lw=2)
plt.plot(xx, ReLU(xx), c='b', label='ReLU', linestyle='-', lw=3)
plt.plot(xx, step(xx), c='k', label='step', linestyle='-.', lw=3)
plt.xticks(size=14)
plt.yticks(size=14)
plt.legend(fontsize=14)
plt.show()
```

### 評価　その2


```python
# 評価処理 (ReLU関数対応版)
from sklearn.metrics import accuracy_score

def evaluate2(x_test, y_test, y_test_one, V, W):
    b1_test = np.insert(ReLU(x_test @ V), 0, 1, axis=1)
    yp_test_one = softmax(b1_test @ W)
    yp_test = np.argmax(yp_test_one, axis=1)
    loss = cross_entropy(y_test_one, yp_test_one)
    score = accuracy_score(y_test, yp_test)
    return score, loss   
```

### 初期化処理


```python
# 変数初期宣言 重み行列の初期化方式変更
# 隠れ層のノード数
H = 128
H1 = H + 1
# M: 訓練用系列データ総数
M  = x_train.shape[0]

# D: 入力データ次元数
D = x_train.shape[1]

# N: 分類クラス数
N = y_train_one.shape[1]

# 機械学習パラメータ
alpha = 0.01
nb_epoch = 100
batch_size = 512
B = batch_size

# 重み行列の初期設定
V = np.random.randn(D, H) / np.sqrt(D / 2)
W = np.random.randn(H1, N) / np.sqrt(H1 / 2)

# 評価結果記録用 (損失関数値と精度)
history3 = np.zeros((0, 3))

# ミニバッチ用関数初期化
indexes = Indexes(M, batch_size)

# 繰り返し回数カウンタ初期化
epoch = 0
```

### メイン処理


```python
# メイン処理 (シグモイド関数をLeRU関数に変更)
while epoch < nb_epoch:
    
    # 学習対象の選択(ミニバッチ学習法)
    index, next_flag = indexes.next_index()
    x, yt = x_train[index], y_train_one[index]

    # 予測値計算 (順伝播) 
    a = x @ V                         # (10.6.3)
    b = ReLU(a)                       # (10.6.4) ReLU化
    b1 = np.insert(b, 0, 1, axis=1)   # ダミー変数の追加 
    u = b1 @ W                        # (10.6.5)   
    yp = softmax(u)                   # (10.6.6)
    
    # 誤差計算 
    yd = yp - yt                      # (10.6.7)
    bd = step(a) * (yd @ W[1:].T)     #(10.6.8) ReLU化

    # 勾配計算
    W = W - alpha * (b1.T @ yd) / B   # (10.6.9)
    V = V - alpha * (x.T @ bd) / B    # (10.6.10)

    if next_flag: # 1epoch 終了後の処理
        score, loss = evaluate2(
            x_test, y_test, y_test_one, V, W)
        history3 = np.vstack((history3, 
            np.array([epoch, loss, score])))
        print("epoch = %d loss = %f score = %f" 
            % (epoch, loss, score))
        epoch = epoch + 1
```

### 結果確認　その3


```python
#損失関数値と精度の確認
print('初期状態: 損失関数:%f 精度:%f' 
        % (history3[0,1], history3[0,2]))
print('最終状態: 損失関数:%f 精度:%f' 
        % (history3[-1,1], history3[-1,2]))
```


```python
# 学習曲線の表示 (損失関数値)
plt.plot(history3[:,0], history3[:,1])
plt.ylim(0,2.5)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()
```


```python
# 学習曲線の表示 (精度)
plt.plot(history3[:,0], history3[:,2])
plt.ylim(0,1)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()
```


```python
# データ内容の確認
import matplotlib.pyplot as plt
N = 20
np.random.seed(123)
indexes = np.random.choice(y_test.shape[0], N, replace=False)

# x_orgの選択結果表示 (白黒反転)
x_selected = x_test[indexes]
y_selected = y_test[indexes]

# 予測値の計算
b1_test = np.insert(ReLU(x_selected @ V), 0, 1, axis=1)
yp_test_one = softmax(b1_test @ W)
yp_test = np.argmax(yp_test_one, axis=1)

# グラフ表示
plt.figure(figsize=(10, 3))
for i in range(N):
    ax = plt.subplot(2, N/2, i + 1)
    plt.imshow(x_selected[i,1:].reshape(28, 28),cmap='gray_r')
    ax.set_title('%d:%d' % (y_selected[i], yp_test[i]),fontsize=14 )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

## 10.10 隠れ層2階層化

### 評価


```python
# 評価処理 (隠れ層2階層対応版)
from sklearn.metrics import accuracy_score

def evaluate3(x_test, y_test, y_test_one, U, V, W):
    b1_test = np.insert(ReLU(x_test @ U), 0, 1, axis=1)
    d1_test = np.insert(ReLU(b1_test @ V), 0, 1, axis=1)
    yp_test_one = softmax(d1_test @ W)
    yp_test = np.argmax(yp_test_one, axis=1)
    loss = cross_entropy(y_test_one, yp_test_one)
    score = accuracy_score(y_test, yp_test)
    return score, loss   
```

### 初期化処理　その3


```python
# 変数初期宣言 重み行列の初期化方式変更
# 隠れ層のノード数
H = 128
H1 = H + 1
# M: 訓練用系列データ総数
M  = x_train.shape[0]

# D: 入力データ次元数
D = x_train.shape[1]

# N: 分類クラス数
N = y_train_one.shape[1]

# 機械学習パラメータ
alpha = 0.01
nb_epoch = 200
batch_size = 512
B = batch_size

# 重み行列の初期設定
U = np.random.randn(D, H) / np.sqrt(D / 2)
V = np.random.randn(H1, H) / np.sqrt(H1 / 2)
W = np.random.randn(H1, N) / np.sqrt(H1 / 2)

# 評価結果記録用 (損失関数値と精度)
history4 = np.zeros((0, 3))

# ミニバッチ用関数初期化
indexes = Indexes(M, batch_size)

# 繰り返し回数カウンタ初期化
epoch = 0
```

### メイン処理　その4


```python
# メイン処理 (隠れ層2層化)

while epoch < nb_epoch:
    # 学習対象の選択(ミニバッチ学習法)
    index, next_flag = indexes.next_index()
    x, yt = x_train[index], y_train_one[index]
    
    # 予測値計算 (順伝播) 
    a = x @ U                        # (10.6.11)
    b = ReLU(a)                      # (10.6.12)
    b1 = np.insert(b, 0, 1, axis=1)  # ダミー変数の追加 
    c = b1 @ V                       # (10.6.13)
    d = ReLU(c)                      # (10.6.14)
    d1 = np.insert(d, 0, 1, axis=1)  # ダミー変数の追加
    u = d1 @ W                       # (10.6.15)   
    yp = softmax(u)                  # (10.6.16)
    
    # 誤差計算 
    yd = yp - yt                     # (10.6.17)
    dd = step(c) * (yd @ W[1:].T)    # (10.6.18)
    bd = step(a) * (dd @ V[1:].T)    # (10.6.19) 

    # 勾配計算
    W = W - alpha * (d1.T @ yd) / B  # (10.6.20)
    V = V - alpha * (b1.T @ dd) / B  # (10.6.21)
    U = U - alpha * (x.T @ bd) / B   # (10.6.22)

    if next_flag: # 1epoch 終了後の処理
        score, loss = evaluate3(
            x_test, y_test, y_test_one, U, V, W)
        history4 = np.vstack((history4, 
            np.array([epoch, loss, score])))
        print("epoch = %d loss = %f score = %f" 
            % (epoch, loss, score))
        epoch = epoch + 1    
```

### 結果の確認


```python
#損失関数値と精度の確認
print('初期状態: 損失関数:%f 精度:%f' 
    % (history4[1,1], history4[1,2]))
print('最終状態: 損失関数:%f 精度:%f' 
    % (history4[-1,1], history4[-1,2]))
```


```python
# 学習曲線の表示 (損失関数値)
plt.plot(history4[:,0], history4[:,1])
plt.ylim(0,2.5)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()
```


```python
# 学習曲線の表示 (精度)
plt.plot(history4[:,0], history4[:,2])
plt.ylim(0,1)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()
```


```python
# データ内容の確認
import matplotlib.pyplot as plt
N = 20
np.random.seed(123)
indexes = np.random.choice(y_test.shape[0], N, replace=False)

# x_orgの選択結果表示 (白黒反転)
x_selected = x_test[indexes]
y_selected = y_test[indexes]

# 予測値の計算
b1_test = np.insert(ReLU(x_selected @ U), 0, 1, axis=1)
d1_test = np.insert(ReLU(b1_test @ V), 0, 1, axis=1)
yp_test_one = softmax(d1_test @ W)
yp_test = np.argmax(yp_test_one, axis=1)

# グラフ表示
plt.figure(figsize=(10, 3))
for i in range(N):
    ax = plt.subplot(2, N/2, i + 1)
    plt.imshow(x_selected[i,1:].reshape(28, 28),cmap='gray_r')
    ax.set_title('%d:%d' % (y_selected[i], yp_test[i]),fontsize=14 )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
