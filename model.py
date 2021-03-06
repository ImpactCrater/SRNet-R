#! /usr/bin/python
# -*- coding: utf8 -*-

import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell


def Generator(t_image, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    df_dim = 128
    swish = lambda x: tf.nn.swish(x)
    with tf.variable_scope("Generator", reuse=reuse) as vs:
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, name='c0')
        n = GroupNormLayer(n, groups=8, act=None, name='gn0')

        # residual in residual blocks
        temp2 = n
        for k in range(2):
            temp1 = n
            for j in range(4):
                temp0 = n
                for i in range(4):
                    nn = Conv2d(n, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c0/%s_%s_%s' % (k, j, i))
                    nn = Conv2d(nn, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c1/%s_%s_%s' % (k, j, i))
                    nn = ElementwiseLayer([n, nn], tf.add, name='res_add0/%s_%s_%s' % (k, j, i))
                    n = nn

                n = Conv2d(n, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c2/%s_%s' % (k, j))
                n = ElementwiseLayer([temp0, n], tf.add, name='res_add1/%s_%s' % (k, j))

            n = Conv2d(n, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c3/%s' % k)
            n = ElementwiseLayer([temp1, n], tf.add, name='res_add2/%s' % k)

        n = Conv2d(n, df_dim, (3, 3), (1, 1), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='res_c4')
        n = ElementwiseLayer([temp2, n], tf.add, name='res_add3')
        # residual in residual blocks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=swish, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=swish, name='pixelshufflerx2/2')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n


def Encoder(input_images, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    df_dim = 64
    swish = lambda x: tf.nn.swish(x)
    with tf.variable_scope("AE", reuse=reuse):
        n = InputLayer(input_images, name='in')
        n = Conv2d(n, df_dim, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, name='c_e0')
        n = GroupNormLayer(n, groups=4, act=None, name='gn_e0')
        n = Conv2d(n, df_dim * 2, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c_e1')
        n = GroupNormLayer(n, groups=8, act=None, name='gn_e1')
        n = Conv2d(n, df_dim * 4, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c_e2')
        n = GroupNormLayer(n, groups=16, act=None, name='gn_e2')
        n = Conv2d(n, df_dim * 8, (4, 4), (2, 2), act=swish, padding='SAME', W_init=w_init, b_init=b_init, name='c_e3')
        n = GroupNormLayer(n, groups=32, act=None, name='gn_e3')
        return n


def Decoder(n, is_train=False, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None # tf.constant_initializer(value=0.0)
    df_dim = 64
    swish = lambda x: tf.nn.swish(x)
    with tf.variable_scope("AE", reuse=reuse) as vs:
        n = Conv2d(n, df_dim * 8, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c_d0')
        n = GroupNormLayer(n, groups=32, act=None, name='gn_d0')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=swish, name='pixelshufflerx2_0')

        n = Conv2d(n, df_dim * 4, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c_d1')
        n = GroupNormLayer(n, groups=16, act=None, name='gn_d1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=swish, name='pixelshufflerx2_1')

        n = Conv2d(n, df_dim * 2, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c_d2')
        n = GroupNormLayer(n, groups=8, act=None, name='gn_d2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=swish, name='pixelshufflerx2_2')

        n = Conv2d(n, df_dim, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='c_d3')
        n = GroupNormLayer(n, groups=4, act=None, name='gn_d3')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=swish, name='pixelshufflerx2_3')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='c_d4')
        return n


def AE(t_image, is_train=False, reuse=False):
    return Decoder(Encoder(t_image, is_train, reuse), is_train, reuse)
