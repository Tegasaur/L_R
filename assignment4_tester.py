import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test1 = x_test.reshape(x_test.shape[0], 28*28)
new_col = np.ones(10000).reshape(10000,1)
x_test1 =  np.append(x_test1,new_col,axis=1)


def softmax(w,x,i):
    w_sum = 0
    for weight in w:
        w_sum += np.exp(weight.dot(x))
        
    wi = np.exp(w[i].dot(x))

    return wi/w_sum

w = np.genfromtxt('weights.csv',delimiter=',')
y1 = []
for i in range(10000):
	z = []
	for j in range(10):
		z.append(softmax(w,x_test1[i],j))
	z = z.index(max(z))
	y1.append(z)
	
for i in range(10000):
    plt.title(str(y1[i]))
    plt.imshow(x_test[i])
    plt.show()
    


