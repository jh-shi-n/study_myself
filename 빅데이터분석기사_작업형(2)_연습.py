# 데이터 파일 읽기 예제
import pandas as pd
import numpy as np
X_test = pd.read_csv("data/X_test.csv")
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

#사용자 코딩
from sklearn.linear_model import LogisticRegression
cust_id = X_test.loc[:,'cust_id']

x_train_df = X_train.drop(['주구매상품','주구매지점','cust_id'], axis=1)
y_train_df = y_train['gender']
x_test_df = X_test.drop(['주구매상품','주구매지점','cust_id'], axis=1)

x_train_df[['환불금액']] = x_train_df[['환불금액']].fillna(0)
x_test_df[['환불금액']] = x_test_df[['환불금액']].fillna(0)

model = LogisticRegression()
model.fit(x_train_df, y_train_df)

#여자/남자일 확률
pred_probs = model.predict_proba(x_test_df)
