# coding: utf-8
'''
Created on 2018. 4. 6.

@author: Insup Jung
'''

import tensorflow as tf
import numpy as np
from Dataset.mnist import * 
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import control_flow_ops

class ConvNetv2(object):

    def __init__(self):
        
        
        learning_rate = 0.0001
        training_epochs = 20
        batch_size = 100
        display_step = 10
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        
        with tf.Graph().as_default():            
            sess = tf.Session()
            x = tf.placeholder(tf.float32, shape=[None, 784])
            t = tf.placeholder(tf.float32, shape=[None, 10])
            self.keep_prob = tf.placeholder(tf.float32)
            keep_prob = self.keep_prob
            self.phase_train = tf.placeholder(tf.bool)
            phase_train = self.phase_train
            output = self.inference(x, keep_prob, phase_train)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            cost = self.loss(output, t)
            train_op = self.training(cost, global_step, learning_rate)
            eval_op = self.evaluate(output, t)
            #summary_op = tf.summary.merge_all()
            #saver = tf.train.Saver()
            
            #summary_writer = tf.summary.FileWriter("board/sample", graph_def=sess.graph_def)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            
            for epoch in range(training_epochs):
                total_batch = int(mnist.train.num_examples/batch_size)
                avg_cost = 0
                for i in range(100):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    feed_dict = {x : batch_x, t : batch_y, keep_prob : 1.0, phase_train : True}
                    sess.run(train_op, feed_dict=feed_dict)
                    accuracy = sess.run(eval_op, feed_dict= feed_dict)
                    #minibatch_cost = sess.run(cost, feed_dict=feed_dict)
                    #avg_cost += minibatch_cost/total_batch
                    if i%display_step==0:
                        
                        accuracy = sess.run(eval_op, feed_dict=feed_dict)
                        print('step', i+epoch*100, 'training accuracy', accuracy)
                        #val_feed_dict = {x : mnist.validation.images, t : mnist.validation.labels, keep_prob : 1.0}
                        #accuracy = sess.run(eval_op, feed_dict=val_feed_dict)
                        #summary_str = sess.run(summary_op, feed_dict=feed_dict)
                        #summary_writer.add_summary(summary_str, sess.run(global_step))
                        #saver.save(sess, "logistic_logs/model-checkpoint", global_step=global_step)
                        #print("Validation Accuracy:", (accuracy))
                    
                #if i%display_step==0 :
                    
            
        print("Optimization Finished!")
                   

        batch_size = 50
        batch_num = int(mnist.test._num_examples / batch_size)
        test_accuracy = 0
        
        for i in range(batch_num):
            batch = mnist.test.next_batch(batch_size)
            test_accuracy += sess.run(eval_op, feed_dict={x:batch[0], t:batch[1], keep_prob : 1.0, phase_train : False})        
        test_accuracy /= batch_num
        print("test accuracy %g"%test_accuracy) #테스트 이미지 확인
        sess.close()        
        
    def conv2d(self, input, weight_shape, bias_shape, phase_train):
        weightX = weight_shape[0]*weight_shape[1]*weight_shape[2] #여기를 이상하게 줬더니 엄청 고생함
        
        weight_init = tf.random_normal_initializer(stddev=(2.0/weightX)**0.5) #왜 stddev를 이렇게 주는지 확인이 필요하다
        W = tf.get_variable("W", weight_shape, initializer=weight_init)
        bias_init = tf.constant_initializer(value = 0)
        b = tf.get_variable("b", bias_shape, initializer=bias_init)
        conv_out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME')
        logits = tf.nn.bias_add(conv_out, b)
        return tf.nn.relu(self.conv_batch_norm(logits, weight_shape[3], phase_train))
    
    def max_pool(self, input, k=2):
        return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    
    def conv_batch_norm(self, x, n_out, phase_train):
        beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
        gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)
        
        beta = tf.get_variable("beta", [n_out], initializer=beta_init)
        gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)
        
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = control_flow_ops.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var))
        normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, True)
        
        return normed
    
    def layer(self, input, weight_shape, bias_shape, phase_train):
        weight_stddev = (2.0/weight_shape[0])**0.5
        w_init = tf.random_normal_initializer(stddev=weight_stddev)
        bias_init = tf.constant_initializer(value=0)

        W = tf.get_variable("W", weight_shape, initializer=w_init)
        b = tf.get_variable("b", bias_shape, initializer=bias_init)
        logits = tf.matmul(input, W)+b
        

        return tf.nn.relu(self.layer_batch_norm(logits, weight_shape[1], phase_train))
    
    def layer_batch_norm(self, x, n_out, phase_train):
        beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
        gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)
        
        beta = tf.get_variable("beta", [n_out], initializer=beta_init)
        gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)
        
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = control_flow_ops.cond(phase_train, mean_var_with_update, lambda: (ema_mean, ema_var))
        x_r = tf.reshape(x, [-1, 1, 1, n_out])
        print('n_out', n_out)
        normed = tf.nn.batch_norm_with_global_normalization(x_r, mean, var, beta, gamma, 1e-3, True)
        
        return tf.reshape(normed, [-1, n_out])
    
    def inference(self, x, keep_prob, phase_train):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        with tf.variable_scope("conv_1"):
            conv_1 = self.conv2d(x, [5, 5, 1, 32], [32], phase_train) #[가로, 세로, 입력수, 출력수] 입력수는 1이다. 채널이 1(흑백)이기 때문
            pool_1 = self.max_pool(conv_1)
        
        with tf.variable_scope("conv_2"):
            conv_2 = self.conv2d(pool_1, [5, 5, 32, 64], [64], phase_train)
            pool_2 = self.max_pool(conv_2)
        
        with tf.variable_scope("fc"):
            pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64]) #4차원 배열을 2차원 배열로 만들어줌
            fc_1 = self.layer(pool_2_flat, [7*7*64, 1024], [1024], phase_train)
            fc_1_drop = tf.nn.dropout(fc_1, keep_prob)
            
        with tf.variable_scope("output"):
            output = self.lastlayer(fc_1_drop, [1024, 10], [10])
        
        return output
        
    
    def lastlayer(self, input, weight_shape, bias_shape):
        weight_stddev = (2.0/weight_shape[0])**0.5
        w_init = tf.random_normal_initializer(stddev=0.1)
        bias_init = tf.constant_initializer(value=0)
        W1 = tf.get_variable("W1", weight_shape, initializer=w_init)
        b1 = tf.get_variable("b1", bias_shape, initializer=bias_init)
        return tf.nn.softmax(tf.matmul(input, W1)+b1)        #softmax를 사용하지 않았더니 엄청 고생함
    
    def loss(self, output, t):
        loss = tf.reduce_mean(-tf.reduce_sum(t*tf.log(tf.clip_by_value(output, 1e-10, 1.0)), reduction_indices=[1]))
        return loss
    
    def evaluate(self, output, t):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy
    
    def training(self, cost, global_step, learning_rate):
        tf.summary.scalar("cost", cost)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(cost)
        
        return train_op
        