# Quiz :
# mtcars 데이터의 특정 컬럼 추출 후, MinMaxScaler 수행.
# 수행한 값 중 0.5보다 높은 값 몇개인지 Count

# 기본 제공 코드
import pandas as pd
a = pd.read_csv('data/mtcars.csv', index_col=0)

# 사용자 코딩
from sklearn.preprocessing  import MinMaxScaler
crop = a[['qsec']] # 특정 컬럼 추출
scaler = MinMaxScaler() #Scaler 준비

crop_scaled = scaler.fit_transform(crop)

print(len(crop_scaled[crop_scaled>0.5]))
