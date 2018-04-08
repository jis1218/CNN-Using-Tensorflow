# coding: utf-8
'''
Created on 2018. 4. 6.

@author: Insup Jung
'''

import tensorflow as tf
import numpy as np
from Dataset.mnist import * 
from tensorflow.examples.tutorials.mnist import input_data

class ConvNetv2(object):

    def __init__(self):
        
        
        learning_rate = 0.1
        training_epochs = 10
        batch_size = 100
        display_step = 10
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        
        with tf.Graph().as_default():            
            x = tf.placeholder(tf.float32, shape=[None, 784])
            t = tf.placeholder(tf.float32, shape=[None, 10])
            self.keep_prob = tf.placeholder(tf.float32)
            keep_prob = self.keep_prob
            output = self.inference(x, keep_prob)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            cost = self.loss(output, t)
            train_op = self.training(cost, global_step, learning_rate)
            eval_op = self.evaluate(output, t)
            summary_op = tf.summary.merge_all()
            saver = tf.train.Saver()
            sess = tf.Session()
            summary_writer = tf.summary.FileWriter("board/sample", graph_def=sess.graph_def)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            
            for epoch in range(training_epochs):
                total_batch = int(mnist.train.num_examples/batch_size)
                avg_cost = 0
                for i in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    feed_dict = {x : batch_x, t : batch_y, keep_prob : 1.0}
                    sess.run(train_op, feed_dict=feed_dict)
                    accuracy = sess.run(eval_op, feed_dict= feed_dict)
                    minibatch_cost = sess.run(cost, feed_dict=feed_dict)
                    avg_cost += minibatch_cost/total_batch
                    if i%display_step==0:
                        print('step', i, 'training accuracy', accuracy)
                    
                if epoch%display_step==0 :
                    accuracy = sess.run(eval_op, feed_dict=feed_dict)
                    print(accuracy)
                    val_feed_dict = {x : mnist.validation.images, t : mnist.validation.labels, keep_prob : 1.0}
                    accuracy = sess.run(eval_op, feed_dict=val_feed_dict)
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, sess.run(global_step))
                    saver.save(sess, "logistic_logs/model-checkpoint", global_step=global_step)
                    print("Validation Accuracy:", (accuracy))
            
        print("Optimization Finished!")
                   
        test_feed_dict = {x : mnist.test.images, t : mnist.test.labels, self.keep_prob : 1.0}
        accuracy = sess.run(eval_op, feed_dict = test_feed_dict )
        print("Test Accuracy:", accuracy)
        sess.close()        
        
    def conv2d(self, input, weight_shape, bias_shape):
        weightX = weight_shape[0]*weight_shape[1]*weight_shape[2] #여기를 이상하게 줬더니 엄청 고생함
        
        weight_init = tf.random_normal_initializer(stddev=(2.0/weightX)**0.5) #왜 stddev를 이렇게 주는지 확인이 필요하다

        W = tf.get_variable("W", weight_shape, initializer=weight_init)
        bias_init = tf.constant_initializer(value = 0)
        b = tf.get_variable("b", bias_shape, initializer=bias_init)
        conv_out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(tf.nn.bias_add(conv_out, b))
    
    def max_pool(self, input, k=2):
        return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    
    def inference(self, x, keep_prob):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        with tf.variable_scope("conv_1"):
            conv_1 = self.conv2d(x, [5, 5, 1, 32], [32]) #[가로, 세로, 입력수, 출력수] 입력수는 1이다. 채널이 1(흑백)이기 때문
            pool_1 = self.max_pool(conv_1)
        
        with tf.variable_scope("conv_2"):
            conv_2 = self.conv2d(pool_1, [5, 5, 32, 64], [64])
            pool_2 = self.max_pool(conv_2)
        
        with tf.variable_scope("fc"):
            pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64]) #4차원 배열을 2차원 배열로 만들어줌
            fc_1 = self.layer(pool_2_flat, [7*7*64, 1024], [1024])

            fc_1_drop = tf.nn.dropout(fc_1, keep_prob)
            
        with tf.variable_scope("output"):
            output = self.lastlayer(fc_1_drop, [1024, 10], [10])
        
        return output
            
    def layer(self, input, weight_shape, bias_shape):
        weight_stddev = (2.0/weight_shape[0])**0.5
        w_init = tf.random_normal_initializer(stddev=weight_stddev)
        bias_init = tf.constant_initializer(value=0)
        W = tf.get_variable("W", weight_shape, initializer=w_init)
        b = tf.get_variable("b", bias_shape, initializer=bias_init)
        return tf.nn.relu(tf.matmul(input, W)+b)  
    
    def lastlayer(self, input, weight_shape, bias_shape):
        weight_stddev = (2.0/weight_shape[0])**0.5
        w_init = tf.random_normal_initializer(stddev=weight_stddev)
        bias_init = tf.constant_initializer(value=0)
        W = tf.get_variable("W", weight_shape, initializer=w_init)
        b = tf.get_variable("b", bias_shape, initializer=bias_init)
        return tf.nn.softmax(tf.matmul(input, W)+b)        #softmax를 사용하지 않았더니 엄청 고생함
    
    def loss(self, output, t):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=output)
        loss = tf.reduce_mean(cross_entropy)
        return loss
    
    def evaluate(self, output, t):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy
    
    def training(self, cost, global_step, learning_rate):
        tf.summary.scalar("cost", cost)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(cost, global_step = global_step)
        return train_op
        