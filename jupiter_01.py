import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing  # カリフォルニア住宅
sns.set_style('whitegrid')

%precision 3
%matplotlib inline

#####
# scikit-learnからカリフォルニア住宅価格のデータセットを読み込んで、
# pandasのデータフレームに格納
#####
housing = fetch_california_housing()

dataset = pd.DataFrame(housing.data, columns=housing.feature_names)
dataset['Price'] = housing.target

# 【実行】
# dataset.head()

#####
# 列ごとにそれぞれどのようなデータ型なのか、欠損があるか、データの数などを、まとめて確認
#####
# 【実行】
# dataset.info()

###
# 欠損値の数のみを知りたい場合は、isnull()に合計値を出し
###
# 【実行】
# dataset.isnull().sum()

###
# 基本統計量を確認の確認（基本統計量とは、平均値や標準偏差といったデータの特徴を代表する値のこと）
###
# 【実行】
# dataset.describe()

#####
# データをヒストグラムにて描画
#####
# 【実行】
# dataset.hist(bins=50, figsize=(15, 13))

#####
# データ同士の相関図を見る。
# 相関図とは、縦軸と横軸にそれぞれ各変数を取ることで、２つの変数間の関係性を見るためのもの。
plt.figure(figsize=(20, 15))
sns.pairplot(dataset.drop(['Latitude', 'Longitude'], axis=1))


####################################################################
#  データの分割
####################################################################

# 目的変数
# 住宅価格を予測する場合、住宅の面積、部屋の数、所在地などが説明変数

# 説明変数
# 住宅価格を予測する場合、実際の住宅価格が目的変数

X = dataset.drop(['Price'], axis=1)  # 説明変数
y = dataset['Price']  # 目的変数
