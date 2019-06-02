# -*- coding:utf-8 -*-
import time
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_float('alpha', 0.005, '')

flags.DEFINE_float('learning_base', 0.0001, '')
flags.DEFINE_integer('decay_step', 15000, '')
flags.DEFINE_float('max_grad_norm', 5, '')
flags.DEFINE_float('learning_rate_decay', 0.99, '')
flags.DEFINE_float('moving_average_decay', 0.99, '')
flags.DEFINE_float('beta', 0.001, '')


class Moptimize(object):
    def __init__(self, model_train):
        with tf.name_scope('Opt'):

            self.global_step = tf.Variable(
                0, name="global_step", trainable=False)
            self.learning_rate = tf.train.exponential_decay(
                FLAGS.learning_base,
                self.global_step,
                FLAGS.decay_step,
                FLAGS.learning_rate_decay,
                staircase=True)
            #self.learning_rate = FLAGS.learning_base
            self.optimizer = tf.train.AdamOptimizer(
                self.learning_rate)
            self.tvars = tf.trainable_variables()
            self.grads, _ = tf.clip_by_global_norm(
                tf.gradients(model_train.loss, self.tvars),
                FLAGS.max_grad_norm)
            self.apply_gradient_op = self.optimizer.apply_gradients(
                zip(self.grads, self.tvars), global_step=self.global_step)
            self.variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, self.global_step)
            self.variables_averages_op = self.variable_averages.apply(
                tf.trainable_variables())
            with tf.control_dependencies(
                [self.apply_gradient_op, self.variables_averages_op]):
                self.train_op = tf.no_op(name='train')
            #return global_step, train_op, learning_rate, grads, tvars
            self.summary = list(self._s())
    def _s(self):
        for g, v in zip(self.grads, self.tvars):
            grad_hist_summary = tf.summary.histogram(
                "{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.histogram(
                "{}/variable/hist".format(v.name), v)
            yield grad_hist_summary
            yield sparsity_summary
