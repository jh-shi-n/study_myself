#패키지 불러오기
import numpy as np

#Class 설정
class Variable:
    def __init__(self, data):
        self.data = data

#Array
data = np.array(1.0)
x = Variable(data)
print(x.data)


x.data = np.array(2.0)
print(x.data)
