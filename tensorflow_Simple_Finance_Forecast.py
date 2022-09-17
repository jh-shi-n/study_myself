from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Flatten
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import tensorflow as tf
tf.set_random_seed(777)

# 데이터 분할 함수
def split(data):
    train_size = int(len(data)*0.6)
    val_size = int(len(data)*0.8)
    data_train = data[:train_size]
    data_val = data[train_size:val_size]
    data_test = data[val_size:]
    return data_train, data_val, data_test

def time_step(data, ts, duration):
    dataX = []
    dataY = []
    
    #lag = 5 고정()
    for i in range(len(data)-ts-(6-1)-(duration-1)):
        dataX.append(data[i:i+ts])
        dataY.append(data[i+ts+6-1:i+ts+6-1+duration])
        
    dataX = np.array(dataX)
    dataY = np.array(dataY)  

    return dataX, dataY

def solo_timestep(data, ts, duration):
    dataX = []
    
    #lag = 5 고정()
    for i in range(len(data)-ts-(6-1)-(duration-1)):
        dataX.append(data[i:i+ts])
        
    dataX = np.array(dataX)

    return dataX

#데이터 불러오기 및 Sort
data = pd.read_csv("C:/Projects/주식데이터_5년.csv")
data_open = data['Open'][1:]
data_close = data['Close'][1:]

#준비, 3일치 데이터를 한번에 input하면서 
gainloss = []
time_select = 3

for i in range(len(data['Volume'])-1):
    if data['Volume'][i+1] > data['Volume'][i]:
        gainloss.append(1)
    elif data['Volume'][i+1] < data['Volume'][i]:
        gainloss.append(0)

dt_o_train, dt_o_val, dt_o_test = split(data_open)
dt_c_train, dt_c_val, dt_c_test = split(data_close)
gl_train, gl_val, gl_test=split(gainloss)

#메인 X,y 설정
x_train, y_train = time_step(dt_c_train,time_select,3)
x_val, y_val = time_step(dt_c_val,time_select,3)
x_test, y_test = time_step(dt_c_test,time_select,3)

#추가데이터 // 시가 데이터
x_o_train = solo_timestep(dt_o_train,time_select,3)
x_o_val = solo_timestep(dt_o_val,time_select,3)
x_o_test = solo_timestep(dt_o_test,time_select,3)

#추가 데이터 // 등락 데이터
t_gl_train = solo_timestep(gl_train,time_select,3)
t_gl_val = solo_timestep(gl_val,time_select,3)
t_gl_test = solo_timestep(gl_test,time_select,3)

#데이터 전처리(2d shape)
x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1],1)
x_val = x_val.reshape((x_val.shape[0],1,x_val.shape[1],1))
x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[1],1))

x_o_train = x_train.reshape(x_o_train.shape[0],1,x_o_train.shape[1],1)
x_o_val = x_val.reshape((x_o_val.shape[0],1,x_o_val.shape[1],1))
x_o_test = x_test.reshape((x_o_test.shape[0],1,x_o_test.shape[1],1))

t_gl_train = t_gl_train.reshape(t_gl_train.shape[0],1,t_gl_train.shape[1],1)
t_gl_val = t_gl_val.reshape(t_gl_val.shape[0],1,t_gl_val.shape[1],1)
t_gl_test = t_gl_test.reshape(t_gl_test.shape[0],1,t_gl_test.shape[1],1)

#데이터 통합
X_train= np.concatenate([x_train,x_o_train,t_gl_train], axis=-1)
X_val= np.concatenate([x_val,x_o_val,t_gl_val], axis=-1)
X_test= np.concatenate([x_test,x_o_test,t_gl_test], axis=-1)

#모델 (CNN 2D > Flatten > DNN)
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(1,2), activation='relu',padding='same', input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3])))
model.add(Conv2D(filters=32, kernel_size=(1,3), activation='relu',padding='same'))
model.add(Conv2D(filters=16, kernel_size=(1,3), activation='relu',padding='valid'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(3))
model.compile(loss='mse', optimizer='adam')

# 조기종료 적용
early_stopping = EarlyStopping(monitor='val_loss', patience = 10)
model.fit(X_train, y_train, epochs=200, batch_size=4, shuffle=False, validation_data=(X_val, y_val), callbacks=[early_stopping])

y_predict = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_predict, y_test))

print(rmse)

# Date : 종가 예측 값 / 종가 실제 값
# 2020/12/22 : 12614.9   / 12807.9
# 2020/12/23 : 12574.535 / 12771.11 
# 2020/12/24 : 12579.771 / 12804.73

# Model RMSE : 476.70
# 실제 값은 차이가 많이 나지만, 종가의 흐름은 예측값과 실제값이 어느정도 비슷하다는 것을 확인 (낮아졌다 소폭 상승)
