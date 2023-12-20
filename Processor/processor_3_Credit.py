# ライブラリ・データセットのインポート
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

FILE_PATH = '/Users/ootsuka/Desktop/プログラミング/Kaggle/Home_Credit_Default_Risk/input/'
OUTPUT_DIR = '/Users/ootsuka/Desktop/プログラミング/Kaggle/Home_Credit_Default_Risk/output/'

train = pd.read_csv(FILE_PATH + 'train.csv')
test = pd.read_csv(FILE_PATH + 'test.csv')

df = pd.concat([train, test], axis=0)
df.reset_index(inplace=True)

# 結合したことでfloatになった"TARGET"をIntに変換
df["TARGET"] = df["TARGET"].astype("Int64")

# 前処理
# 欠損値の補完
def missing_values(df):
    # 0日は欠損値扱いする
    df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True) # set null value

    return df

# 外れ値の削除
def outlier(df):
    # 'AMT_INCOME_TOTAL' が 20000000 以上の場合、欠損値 (NaN) に置き換える
    df.loc[df['AMT_INCOME_TOTAL'] >= 20000000, 'AMT_INCOME_TOTAL'] = np.nan
    # エラーの日数のデータは欠損値に置き換える
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    return df

# 特徴量エンジニアリング
# 特徴量の作成
def create_new_features(df):
    # AMT（お金関係） に関する特徴量（Amount？）

    # クレジット額（‘AMT_CREDIT’） /
    # 定期的に支払うローン("AMT_ANNUITY") # 年金支払額を借入金で割る
    df["Credit_Annuity_Ratio"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"] # 重要
    # 融資対象となる商品の価格 # 借入金をローン対象額で割る
    df["Credit_Goods_Ratio"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]

    # 収入額（’AMT_INCOME_TOTAL’） /
    # 定期的に支払うローン # 年金支払額を所得で割る
    df["Income_Annuity_Ratio"] = df["AMT_INCOME_TOTAL"] / df["AMT_ANNUITY"]
    # ローン額(’AMT_CREDIT’) 借入金を所得で割る
    df["Income_Credit_Ratio"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    # 家族の人数と収入
    df["Income_Per_Person"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    # 雇用日(’DAYS_EMPLOYED’) / # 総所得金額を就労期間で割る
    df['Income_Employed_Ratio'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
    # 誕生日('DAYS_BIRTH') /
    df['Income_Birth_Ratio'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']

    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['INCOME_PER_FAMILY_MEMBER'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['INCOME_TO_REGISTRATION_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_REGISTRATION']
    df['INCOME_SUB_ANNUITY_PER_MONTH'] = df['AMT_INCOME_TOTAL'] / 12. - df['AMT_ANNUITY']

    # 定期的に支払うローン("AMT_ANNUITY")
    df['ANNUITY_TO_GOODS_PRICE_RATIO'] = df['AMT_ANNUITY'] / df['AMT_GOODS_PRICE']
    df['ANNUITY_TO_AGE_RATIO'] = df['AMT_ANNUITY'] / df['DAYS_BIRTH']
    df['ANNUITY_TO_EMPLOYED_RATIO'] = df['AMT_ANNUITY'] / df['DAYS_EMPLOYED']
    df['ANNUITY_TO_REGISTRATION_RATIO'] = df['AMT_ANNUITY'] / df['DAYS_REGISTRATION']
    df['ANNUITY_TO_ID_PUBLISH_RATIO'] = df['AMT_ANNUITY'] / df['DAYS_ID_PUBLISH']

    # ローン額(’AMT_CREDIT’)
    df['CREDIT_PER_FAMILY_MEMBER'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']
    df['CREDIT_PER_CHILD'] = df['AMT_CREDIT'] / (1 + df['CNT_CHILDREN'])
    df['CREDIT_TO_GOODS_PRICE_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['CREDIT_TO_AGE_RATIO'] = df['AMT_CREDIT'] / df['DAYS_BIRTH']

    # 申請の1時間前にクライエントに問い合わせた回数('AMT_REQ_CREDIT_BUREAU_HOUR')
    df['ENQUIRY_CREDIT_BUREAU_HOUR_TO_DAY'] = df['AMT_REQ_CREDIT_BUREAU_HOUR'] * 24
    df['ENQUIRY_CREDIT_BUREAU_MONTH_TO_QUARTER_RATIO'] = df['AMT_REQ_CREDIT_BUREAU_MON'] / df['AMT_REQ_CREDIT_BUREAU_QRT']

    # ローンが与えられる商品の価格
    df['GOODS_PRICE_TO_AGE_RATIO'] = df['AMT_GOODS_PRICE'] / df['DAYS_BIRTH']
    df['GOODS_PRICE_TO_EMPLOYED_RATIO'] = df['AMT_GOODS_PRICE'] / df['DAYS_EMPLOYED']
    df['GOODS_PRICE_TO_INCOME_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL']
    # ローン対象金額から借入金を引く
    df["DOWNPAYMENT"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]

    # 日付に関する特徴量

    # 年齢
    # 申し込み時点の年齢
    df["Age"] = -df["DAYS_BIRTH"] / 365

    # 雇用日(’DAYS_EMPLOYED’) /
    # 誕生日 # 就労期間を年齢で割る
    df['Days_Employed_Birth_Ratio'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['EMPLOYED_SUB_AGE'] = df['DAYS_EMPLOYED'] - df['DAYS_BIRTH']
    # 車の所持年数（’OWN_CAR_AGE’）
    df['Car_Employed_Ratio'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['DAYS_BIRTH_365_OWN_CAR_AGE']= (df['DAYS_BIRTH']/ 365)- df['OWN_CAR_AGE']

    # 誕生日('DAYS_BIRTH') /
    # ID変更日('DAYS_ID_PUBLISH')
    df['Id_Birth_Ratio'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    # 車の所持年数（’OWN_CAR_AGE’）
    df['Car_Birth_Ratio'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    # 電話の変更日('DAYS_LAST_PHONE_CHANGE')
    df['Phone_Birth_Ratio'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']

    # ‘Ext_Source_*’（外部データ）に関する特徴量
    # 外部データの最小値、最大値、平均、中央値、分散を計算し、新たな特徴量を作成
    for function_name in ['min', 'max', 'mean', 'nanmedian', 'var']:
        feature_name = 'EXT_Sources_{}'.format(function_name.upper())
        df[feature_name] = eval('np.{}'.format(function_name))(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)


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
    X = X.drop(['TARGET'], axis=1) # 目的変数を指定する
    y = df['TARGET'] # 目的変数を指定する

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
        # 重要度 0
        'REG_REGION_NOT_WORK_REGION', 'DAYS_LAST_PHONE_CHANGE', 'ANNUITY_TO_ID_PUBLISH_RATIO', 
        'REG_REGION_NOT_LIVE_REGION', 'REGION_RATING_CLIENT', 'ENQUIRY_CREDIT_BUREAU_HOUR_TO_DAY',
        'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'CREDIT_TO_AGE_RATIO', 'DAYS_ID_PUBLISH', 'GOODS_PRICE_TO_AGE_RATIO',
        'DAYS_EMPLOYED', 'DAYS_BIRTH', 'DOWNPAYMENT', 'GOODS_PRICE_TO_EMPLOYED_RATIO', 'Income_Birth_Ratio',
        'INCOME_PER_FAMILY_MEMBER', 'DAYS_BIRTH_365_OWN_CAR_AGE', 'INCOME_TO_REGISTRATION_RATIO', 'CREDIT_TO_GOODS_PRICE_RATIO',
        'ANNUITY_TO_EMPLOYED_RATIO', 'Income_Employed_Ratio',
        # 重要度 1~8
        'FLAG_EMAIL', 'REG_CITY_NOT_WORK_CITY', 'FLAG_CONT_MOBILE', 'LIVE_REGION_NOT_WORK_REGION', 'FLAG_OWN_CAR', # 1
        'LIVE_CITY_NOT_WORK_CITY', 'DAYS_REGISTRATION', # 2
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'FLAG_PHONE', 'CNT_FAM_MEMBERS', # 3
        'AMT_REQ_CREDIT_BUREAU_MON', 'CNT_CHILDREN', # 4
        'FLAG_OWN_REALTY', # 5
        'NAME_TYPE_SUITE', # 6
        'NAME_HOUSING_TYPE', # 8
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

train = df[df.loc[:, 'SK_ID_CURR'] < 171202]
test = df[df.loc[:, 'SK_ID_CURR'] > 171201]

train_x = train.drop(columns=['TARGET', 'SK_ID_CURR'])
train_y = train['TARGET']
test_x = test.drop(columns=['TARGET', 'SK_ID_CURR'])

X = train_x.values
y = train_y.values
y = y.astype(int)

# ID削除
df.drop("SK_ID_CURR", axis=1, inplace=True)

# 初期の方でカラム名でエラー出たので（名残）
# 文字列の特殊文字を削除
df.columns = df.columns.str.replace('[^\w\s]', '').str.replace(':', '')
# 列名の空白をアンダースコアに置き換え
df.columns = df.columns.str.replace(' ', '_')

df.to_csv(OUTPUT_DIR + 'data.csv', index=False)