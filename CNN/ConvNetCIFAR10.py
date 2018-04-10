# coding: utf-8
'''
Created on 2018. 4. 9.

@author: Insup Jung
'''

import tensorflow as tf
import numpy as np
from Dataset.mnist import * 
from tensorflow.examples.tutorials.mnist import input_data

class MyClass(object):

    def __init__(self, params):
        mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
        
        x = tf.placeholder(tf.float32, [None, 784])
        t = tf.placeholder(tf.float32, [None, 10])
        
    
    
    def inference(self, x):
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])
        with tf.variable_scope("first conv"):
            conv_1 = self.conv2d(x_reshape, [5, 5, 1, 32], [32])
            pool_1 = self.max_pool(conv_1)
        
        with tf.variable_scope("second conv"):
            conv_2 = self.conv2d(pool_1, [5, 5, 32, 64], [64])
            pool_2 = self.max_pool(conv_2)
            
        with tf.variable_scope("fc1"):
            
    
    
    def conv2d(self, input, weight_shape, bias_shape):
        weightX = weight_shape[0]*weight_shape[1]*weight_shape[2]
        weight_init = tf.random_normal_initializer(stddev=0.1)
        W = tf.get_variable("W", weight_shape, initializer=weight_init)
        bias_init = tf.constant_initializer(value = 0)
        b = tf.get_variable("b", bias_shape, initializer=bias_init)
        output = tf.nn.conv2d(input, W, [1, 1, 1, 1], 'SAME')
        return output
        
        
    def max_pool(self, output, k=2):
        return tf.nn.max_pool(output, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
        
    def layer(self, x, weight_shape, bias_shape):
        weight_init = tf.random_normal_initializer(stddev=0.1, seed, dtype)
        
        

        