import numpy as np
import matplotlib.pyplot as pl
import random
import keras
from keras.datasets import mnist
import cv2
from math import exp
import time


(x_train, y_train), (x_test, y_test) = mnist.load_data()
images = x_test

x_train = x_train.reshape(x_train.shape[0], 28*28)
new_col = np.ones(60000).reshape(60000,1)
x_train =  np.append(x_train,new_col,axis=1)

x_test = x_test.reshape(x_test.shape[0], 28*28)
new_col = np.ones(10000).reshape(10000,1)
x_test =  np.append(x_test,new_col,axis=1)

target = np.zeros((60000,10))

for items in range(len(y_train)):
    target[items][y_train[items]] = 1

w = 1*(10**-5) * np.ones((10,785))

def softmax(w,x,i):
    w_sum = 0
    for weight in w:
        w_sum += np.exp(weight.dot(x))
        
    wi = np.exp(w[i].dot(x))

    return wi/w_sum


lamda = (10**-10)
J = 0
for j in range (10):
    for i in range(60000):
        J += -target[i][j]*np.log(softmax(w,x_train[i],j))
count = 0



while count<200:
    count+=1
    stand_in = np.zeros((10,785))
    for item in range(10):
        for iteration in range(60000):
            stand_in[item] += ((softmax(w,x_train[iteration],item) - target[iteration][item])*x_train[iteration])

    
    for item in range(10):
        w[item] = w[item]- lamda*stand_in[item]

    J = 0
    for j in range (10):
        for i in range(60000):
            J += -target[i][j]*np.log(softmax(w,x_train[i],j))
    print(J)
y1 = []
for i in range(10000):
	z = []
	for j in range(10):
		z.append(softmax(w,x_test[i],j))
	z = z.index(max(z))
	y1.append(z)











