# ライブラリ・データセットのインポート
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

## 実行時間を調べるために使う
import datetime
import time
import math

start_time = time.time()

def changeHMS(s):
    h = math.floor(s / 3600)
    if h > 0:
        s = s - h * 3600
        indi_h = str(h) + 'h'
    else:
        indi_h = ''
    m = math.floor(s / 60)
    if m > 0:
        indi_m = str(m) + 'm'
    else:
        indi_m = ''
    s = math.floor(s % 60)
    time = indi_h + indi_m + str(s) + 's'
    return time

FILE_PATH = '/Users/ootsuka/Desktop/プログラミング/Kaggle/肝硬変の転帰の多クラス予測/input/'
OUTPUT_DIR = '/Users/ootsuka/Desktop/プログラミング/Kaggle/肝硬変の転帰の多クラス予測/output/'

train = pd.read_csv(FILE_PATH + 'train.csv')
test = pd.read_csv(FILE_PATH + 'test.csv')

target = train['Status']

target_name = str(train.iloc[:, [18]].columns.tolist()) # カラム数-2の値が目的変数

df = pd.concat([train, test], axis=0)
df.reset_index(inplace=True)

# 説明変数をデータ型ごとに代入する
numerical_features = df._get_numeric_data().columns
categorical_features = df.drop(numerical_features, axis=1).columns

# 前処理
# 欠損値の補完
def missing_values(df):

    return df

# 外れ値の削除
def outlier(df):

    return df

# 特徴量エンジニアリング
# 特徴量の作成
def create_new_features(df):
    
    return df

# カテゴリ変数のエンコーディング
# LabelEncoding
def label_encoder(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = df[column].fillna('').astype('str') # 欠損値の補完をする
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])

    return df

# 特徴量の選択
# 特徴量の重要度評価
def feature_importance_evaluation(df):
    # データを対数変換する

    # 訓練データをX(説明変数)とy（目的変数）に分割する
    X = df.select_dtypes(include=['float', 'int'])
    X = X.drop(['Status'], axis=1) # 目的変数を指定する
    y = target # 目的変数を指定する

    for column in X.columns.tolist():
        X[column] = X[column].apply(lambda x: np.log(x + 1))

    # 特徴量の重要度評価
    lgb = LGBMClassifier(
        random_state=42,
    )

    lgb.fit(X, y)
    importance = lgb.feature_importances_

    feature_importance = pd.DataFrame(data=importance, index=X.columns, columns=['importance']) \
        .sort_values(ascending=True, by='importance')

    return feature_importance

# 特徴量の削除
def drop_columns(df):
    drop_list = [
        
    ]
    dropped_df = df.drop(columns=drop_list)

    return dropped_df

# データセットの更新
# 前処理
df = missing_values(df)
df = outlier(df)

# 特徴量エンジニアリング
df = create_new_features(df)
df = drop_columns(df)
df = label_encoder(df)

train = df[df.loc[:, 'id'] < 7905]
test = df[df.loc[:, 'id'] >= 7905]

train_x = train.drop(columns=['Status', 'id'])
train_y = target
test_x = test.drop(columns=['Status', 'id'])

X = train_x.values
y = train_y.values
# y = y.astype(int)

df.head()

# ID削除
df.drop("id", axis=1, inplace=True)

df.to_csv(OUTPUT_DIR + 'data.csv', index=False)

# 確認 (data_import.py)
def file_to_xy(filename):
    data = pd.read_csv(filename, index_col=0)
    print(f'読み込み完了 {filename}')
    train = data[:7905].reset_index(drop=True)
    test = data[7905:].reset_index(drop=True).drop('Status', axis=1)
    # 目的変数と説明変数に分割
    X = train.drop('Status', axis=1)
    y = train['Status'].values
    return data,test,train,X,y

filename = OUTPUT_DIR + 'data.csv'
data,test,train,X,y = file_to_xy(filename)