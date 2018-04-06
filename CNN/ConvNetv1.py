# coding: utf-8
'''
Created on 2018. 3. 29.

@author: Insup Jung
'''

import tensorflow as tf
import numpy as np
from Dataset.mnist import * 
from tensorflow.examples.tutorials.mnist import input_data

class ConvNetv1(object):

    def __init__(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        sess = tf.InteractiveSession()
    
    
        x = tf.placeholder(tf.float32, shape=[None, 784])
        t = tf.placeholder(tf.float32, shape=[None, 10])
        
        x_image = tf.reshape(x, [-1, 28, 28, 1]) #이미지가 2차원 배열이므로 4차원으로 만들어준다. -1이 있는 이유는 다차원 배열의 원소 수가 변환 후에도 똑같이 유지되도록 하기 위해서이다.
        
        # Conv/pool layer
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1) #2차원 배열끼리 합성곱을 실행한다. 필터가 32개이므로 출력이 32인 텐서가 나온다.
        h_pool1 = self.max_pool_2x2(h_conv1) #풀링 계층 실행, 이미지 크기가 [14, 14]가 된다.
        
        
        W_conv2 = self.weight_variable([5, 5, 32, 64]) # 전에서 출력이 32가 나왔으므로 입력이 32가 되고 출력은 64가 된다.
        b_conv2 = self.bias_variable([64])
        
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2)+b_conv2)
        
        h_pool2 = self.max_pool_2x2(h_conv2) #두번째 풀링 계층 실행, 이미지 크기가 [7, 7], 출력은 64가 된다.
        
        
        
        #Fully-Connected Layer (완전 연결 계층) - 1024개의 은닉노드를 가지고 있는 계층을 구성한다.
        
        W_fc1 = self.weight_variable([7*7*64, 1024])
        b_fc1 = self.bias_variable([1024])
        
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
        
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])
        
        y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2)+b_fc2)
        
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(t*tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)), reduction_indices=[1]))
        # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            batch = mnist.train.next_batch(50)
            if i%10==0 :
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], t:batch[1]})
                print("step %d, training accuracy %g"%(i, train_accuracy))
            
            train_step.run(feed_dict={x: batch[0], t: batch[1]}) #학습이미지 확인
                
        print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, t: mnist.test.labels})) #테스트 이미지 확인
        
        # 시간이 상당히 많이 걸린다. 그 이유는 합성곱 과정에서 시간을 많이 잡아먹기 때문
        
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #4차원 텐서를 2차원 배열로 만들어주는 역할 - Computes a 2-D convolution given 4-D `input` and `filter` tensors. padding을 'SAME'으로 하면 출력 크기가 입력과 같게 되도록 0으로 패딩하도록 설정한다. 즉 입력 크기가 [28, 28] 이라면 출력 크기도 [28, 28]이다.
    
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    