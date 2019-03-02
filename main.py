#! /usr/bin/python3
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import math

import tensorflow as tf
import tensorlayer as tl
from model import gen
from utils import *
from config import config, log_config
from PIL import Image

import re
import glob

###====================== HYPER-PARAMETERS ===========================###
## batch size
sample_batch_size = config.TRAIN.sample_batch_size
## Adam
batch_size = config.TRAIN.batch_size
learning_rate = config.TRAIN.learning_rate
beta1 = config.TRAIN.beta1
## train
n_epoch = config.TRAIN.n_epoch
## paths
samples_path = config.samples_path
checkpoint_path = config.checkpoint_path
valid_hr_img_path = config.VALID.hr_img_path
train_hr_img_path = config.TRAIN.hr_img_path
eval_img_name = config.VALID.eval_img_name
eval_img_path = config.VALID.eval_img_path

ni = int(np.sqrt(sample_batch_size))


def load_deep_file_list(path=None, regx='\.npz', printable=True):
    if path == False:
        path = os.getcwd()
    pathStar = path + '**'
    file_list = glob.glob(pathStar, recursive=True)
    return_list = []
    for idx, f in enumerate(file_list):
        if re.search(regx, f):
            fShort = f.replace(path,'')
            return_list.append(fShort)
    if printable:
        print('Match file list = %s' % return_list)
        print('Number of files = %d' % len(return_list))
    return return_list


def train():
    ## create folders to save result images and trained model
    save_dir_gen = samples_path + "gen"
    tl.files.exists_or_mkdir(save_dir_gen)
    tl.files.exists_or_mkdir(checkpoint_path)

    ###====================== PRE-LOAD DATA ===========================###
    valid_hr_img_list = sorted(tl.files.load_file_list(path=valid_hr_img_path, regx='.*.png', printable=False))

    ###========================== DEFINE MODEL ============================###
    ## train inference
    sample_t_image = tf.placeholder('float32', [sample_batch_size, 96, 96, 3], name='sample_t_image_input_to_generator')
    t_image = tf.placeholder('float32', [batch_size, 96, 96, 3], name='t_image_input_to_generator')
    t_target_image = tf.placeholder('float32', [batch_size, 384, 384, 3], name='t_target_image')

    net_g = gen(t_image, is_train=True, reuse=False)

    net_g.print_params(False)
    net_g.print_layers()

    ## test inference
    net_g_test = gen(sample_t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###

    # MSE Loss
    with tf.name_scope('MSE_Loss'):
        mse_loss = tf.reduce_mean(tf.square(t_target_image - net_g.outputs))
        tf.summary.scalar('mse_loss', mse_loss)

    merged = tf.summary.merge_all()

    with tf.variable_scope('learning_rate'):
        learning_rate_var = tf.Variable(learning_rate, trainable=False)

    g_vars = tl.layers.get_variables_with_name('gen', True, True)

    ## Train
    g_optim = tf.train.AdamOptimizer(learning_rate=learning_rate_var, beta1=beta1).minimize(mse_loss, var_list=g_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.variables_initializer(tf.global_variables()))
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_path + 'gen.npz', network=net_g)

    ###============================= TRAINING ===============================###
    sample_imgs = tl.prepro.threading_data(valid_hr_img_list[0:sample_batch_size], fn=get_imgs_fn, path=valid_hr_img_path)
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print('sample HR sub-image:', sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gen + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gen + '/_train_sample_384.png')

    ###========================= train Generator ====================###
    ## fixed learning rate
    sess.run(tf.assign(learning_rate_var, learning_rate))
    print(" ** fixed learning rate: %f" % learning_rate)
    for epoch in range(0, n_epoch + 1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        train_hr_img_list = load_deep_file_list(path=train_hr_img_path, regx='.*.png', printable=False)
        random.shuffle(train_hr_img_list)

        list_length = len(train_hr_img_list)
        print("Number of images: %d" % (list_length))

        if list_length % batch_size != 0:
            train_hr_img_list += train_hr_img_list[0:batch_size - list_length % batch_size:1]

        list_length = len(train_hr_img_list)
        print("Length of list: %d" % (list_length))

        for idx in range(0, list_length, batch_size):
            step_time = time.time()
            b_imgs_list = train_hr_img_list[idx : idx + batch_size]
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=train_hr_img_path)
            b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            b_imgs_384 = tl.prepro.threading_data(b_imgs_384, fn=rescale_m1p1)

            ## update Generator
            errM, _ = sess.run([mse_loss, g_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.10f " % (epoch, n_epoch, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.10f" % (epoch, n_epoch, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != - 1) and (epoch % 1 == 0):
            out = sess.run(net_g_test.outputs, {sample_t_image: sample_imgs_96})
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gen + '/train_%d.png' % epoch)

        ## save model
        if (epoch != - 1) and (epoch % 1 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_path + 'gen.npz'.format(tl.global_flag['mode']), sess=sess)


def evaluate():
    ## create folders to save result images
    save_dir = samples_path + "{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)

    ###========================== DEFINE MODEL ============================###
    valid_lr_img = get_imgs_fn(eval_img_name, eval_img_path) # if you want to test your own image
    valid_lr_img = rescale_m1p1(valid_lr_img)

    size = valid_lr_img.shape
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = gen(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.variables_initializer(tf.global_variables()))
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_path + 'gen.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir + '/valid_gen.png')

    out_bicu = (valid_lr_img + 1) * 127.5 # rescale to [0, 255]
    out_bicu = np.array(Image.fromarray(np.uint8(out_bicu)).resize((size[1] * 4, size[0] * 4), Image.BICUBIC))
    tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'train':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
