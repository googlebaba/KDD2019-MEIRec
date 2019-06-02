# !/usr/bin/python2
# -*- coding: utf-8 -*-
import tensorflow as tf
from utils import get_placeholder, update_placeholder, data_precess 
from MultiRecallSimpV1_exp_U_iq_qi_Q_ui_iq_lstm import Model
#from MultiRecallSimpV1_exp_U_iq_qi_Q_ui_iq_lstm_cnn import Model
from MOptimizer import Moptimize
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
train_path = './train_data.txt'
test_path = './test_data.txt'
test_size = 5000
training_epochs=10


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True  #设置tf模式为按需赠长模式
    sess = tf.Session(config=config)
   
    placeholders = get_placeholder()

    with tf.variable_scope("model", reuse=None):
        model_train = Model()
        model_train.build(placeholders)
    
    with tf.variable_scope("model", reuse=True):
        model_test = Model(is_training=False)
        model_test.build(placeholders)
    with tf.variable_scope("LearningRate"):
        opt = Moptimize(model_train)
    with tf.variable_scope("summary"):
        train_summary_merged = tf.summary.merge([model_train.summary, opt.summary], [])
        test_summary_merged = tf.summary.merge(model_test.summary)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    train_data = data_precess(train_path)
    test_data = data_precess(test_path)
    step = 0
    for epoch in range(training_epochs):
        batches = train_data.batch_iter(batch_size=512)
        for batch in batches:
            step += 1
            feed_dict = update_placeholder(placeholders, batch)
            _, _train_loss, _train_auc = sess.run([opt.train_op, model_train.loss, model_train.AUC_real], feed_dict=feed_dict)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'iter', step, "==============")
            print("   train: auc = ", _train_auc, "loss:", _train_loss)
        if epoch % 2 == 0:
            batches = test_data.batch_iter(batch_size=test_data.data_size)
        for batch in batches:
            feed_dict_test = update_placeholder(placeholders, batch)
            _test_loss, _test_auc = sess.run([model_train.loss, model_train.AUC_real], feed_dict=feed_dict_test)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'iter', step, "==============")
            print("   test: auc = ", _test_auc, "loss:", _test_loss)





