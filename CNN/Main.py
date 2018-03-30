# coding: utf-8
'''
Created on 2018. 3. 29.

@author: Insup Jung
'''
import tensorflow as tf
import numpy as np
from Dataset.mnist import * 
from tensorflow.examples.tutorials.mnist import input_data
if __name__ == '__main__':
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #4차원 텐서를 2차원 배열로 만들어주는 역할 - Computes a 2-D convolution given 4-D `input` and `filter` tensors. padding을 'SAME'으로 하면 출력 크기가 입력과 같게 되도록 0으로 패딩하도록 설정한다.
    
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    
    
    x = tf.placeholder(tf.float32, shape=[None, 784])
    t = tf.placeholder(tf.float32, shape=[None, 10])
    
    x_image = tf.reshape(x, [-1, 28, 28, 1]) #이미지가 2차원 배열이므로 4차원으로 만들어준다. -1이 있는 이유는 다차원 배열의 원소 수가 변환 후에도 똑같이 유지되도록 하기 위해서이다.
    
    # Conv/pool layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #2차원 배열끼리 합성곱을 실행한다.
    h_pool1 = max_pool_2x2(h_conv1) #풀링 계층 실행
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    #Fully-Connected Layer
    
    
    
    
    
    pass

