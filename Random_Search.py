import random
import os
import tensorflow as tf
import models.net_factory as nf
import numpy as np
from data_handler import Data_handler

import random
import sys
from termcolor import  colored
import sqlite3
import datetime
from numba import cuda
now=datetime.datetime.now()
import math
flags = tf.app.flags

flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('num_iter', 10000, 'Total training iterations')
flags.DEFINE_string('model_dir', '/home/mohammad/', 'Trained network dir')
flags.DEFINE_string('data_version', 'kitti2015', 'kitti2012 or kitti2015')
flags.DEFINE_string('data_root', '/home/mohammad/data_scene_flow/training', 'training dataset dir')
flags.DEFINE_string('util_root', '/home/mohammad/data_scene_flow', 'Binary training files dir')
flags.DEFINE_string('net_type', 'win37_dep9', 'Network type: win37_dep9 pr win19_dep9')

flags.DEFINE_integer('eval_size', 200, 'number of evaluation patchs per iteration')
flags.DEFINE_integer('num_tr_img', 160, 'number of training images')
flags.DEFINE_integer('num_val_img', 40, 'number of evaluation images')
flags.DEFINE_integer('patch_size', 37, 'training patch size')
flags.DEFINE_integer('num_val_loc', 50000, 'number of validation locations')
flags.DEFINE_integer('disp_range', 201, 'disparity range')
flags.DEFINE_string('phase', 'train', 'train or evaluate')

FLAGS = flags.FLAGS

np.random.seed(123)
tf.debugging.set_log_device_placement(True)
dhandler = Data_handler(data_version=FLAGS.data_version,
                        data_root=FLAGS.data_root,
                        util_root=FLAGS.util_root,
                        num_tr_img=FLAGS.num_tr_img,
                        num_val_img=FLAGS.num_val_img,
                        num_val_loc=FLAGS.num_val_loc,
                        batch_size=FLAGS.batch_size,
                        patch_size=FLAGS.patch_size,
                        disp_range=FLAGS.disp_range)

if FLAGS.data_version == 'kitti2012':
    num_channels = 1
elif FLAGS.data_version == 'kitti2015':
    num_channels = 3
else:
    sys.exit('data_version should be either kitti2012 or kitti2015')

def Valid_Generator():
    Valids=37
    Valid_Layers=[]
    while(Valids>1):
        #print(Valids)
        x=random.randint(1,(Valids-1)/2)
        #print('x=',x)
        Valid_Layers.append(2*x+1)
        Valids=Valids-(2*x)
    count=0
    sum=0
    for l in Valid_Layers:
        count+=1
        sum+=l
    if sum-count==36:
        Valid_list=[]
        for l in Valid_Layers:
            Valid_list.append(['conv2d',random.choice([32,64]),'valid',l])
        '''print(Valid_list)
        list=[['none',0,'none',0]]*10
        list.extend(Valid_list)
        random.shuffle(list)'''
        return Valid_list


    else:
        raise ValueError('dfbdbnb')
def None_Valid_Generator():
    length=random.randint(0,100)
    None_Valid_List=[]
    layers = [['conv2d', 32, 'same', 3], ['conv2d', 32, 'same', 5], ['conv2d', 32, 'same', 7],
              ['conv2d', 32, 'same', 11], ['batch', 0, 'none', 0], ['conc', 0, 'none', 0]]
    for i in range(length):
        None_Valid_List.append(random.choice(layers))
    return None_Valid_List
def OneD_Generator():
    Valid_list=Valid_Generator()
    list=None_Valid_Generator()
    list.extend(Valid_list)
    random.shuffle(list)


    kernel_sum = 0
    num_node = 0
    for i in list:
        if i[0] == 'conv2d':
            if i[2] == 'valid':
                kernel_sum += i[3]
                num_node += 1
    ex = kernel_sum - num_node
    if ex != 36:

        raise ValueError('this is not appropriante')
    return list
def TwoD_Generator():
    TwoD_list=[]
    TwoD_list.append(OneD_Generator())
    TwoD_list.append(OneD_Generator())
    return TwoD_list
def train(state, number):
    path = FLAGS.model_dir + '/' + str(number)
    if not os.path.exists(path):
        os.makedirs(path)
    tf.reset_default_graph()
    run_meta = tf.RunMetadata()
    g = tf.Graph()
    #cuda.select_device(0)
    strategy=tf.distribute.MirroredStrategy()
    with strategy.scope():
        with g.as_default():
            log=(tf.Session(config=tf.ConfigProto(log_device_placement=True)).list_devices())

            file=open(path+'\\log.txt','a+')
            file.write(str(log))
            file.close()





            limage = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, num_channels], name='limage')
            rimage = tf.placeholder(tf.float32,
                                    [None, FLAGS.patch_size, FLAGS.patch_size + FLAGS.disp_range - 1, num_channels],
                                    name='rimage')
            targets = tf.placeholder(tf.float32, [None, FLAGS.disp_range], name='targets')

            snet = nf.create(limage, rimage, targets, state, FLAGS.net_type)

            loss = snet['loss']
            train_step = snet['train_step']
            session = tf.InteractiveSession()
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)

            acc_loss = tf.placeholder(tf.float32, shape=())
            loss_summary = tf.summary.scalar('loss', acc_loss)
            train_writer = tf.summary.FileWriter(path + '/training', g)

            saver = tf.train.Saver(max_to_keep=1)
            losses = []
            summary_index = 1
            lrate = 1e-2

            for it in range(1, FLAGS.num_iter):
                lpatch, rpatch, patch_targets = dhandler.next_batch()

                train_dict = {limage: lpatch, rimage: rpatch, targets: patch_targets,
                              snet['is_training']: True, snet['lrate']: lrate}
                _, mini_loss = session.run([train_step, loss], feed_dict=train_dict)
                losses.append(mini_loss)

                if it % 10 == 0:
                    print('Loss at step: %d: %.6f' % (it, mini_loss)) #please us me later
                    saver.save(session, os.path.join(path, 'model.ckpt'), global_step=snet['global_step'])
                    train_summary = session.run(loss_summary,
                                                feed_dict={acc_loss: np.mean(losses)})
                    train_writer.add_summary(train_summary, summary_index)
                    summary_index += 1
                    train_writer.flush()
                    losses = []

                if it == 24000:
                    lrate = lrate / 5.
                elif it > 24000 and (it - 24000) % 8000 == 0:
                    lrate = lrate / 5.
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
        if flops is not None:
            t_flops=flops.total_float_ops
            print('wetwyy', t_flops)
    #cuda.select_device(0)

    #cuda.close()
    return t_flops


def evaluate(state,number):
    lpatch, rpatch, patch_targets = dhandler.evaluate()
    labels = np.argmax(patch_targets, axis=1)
    path = FLAGS.model_dir + '/' + str(number)
    print('path=',path)


    #with tf.device('/gpu:0'):
    with tf.Session() as session:
        limage = tf.placeholder(tf.float32, [None, FLAGS.patch_size, FLAGS.patch_size, num_channels], name='limage')
        rimage = tf.placeholder(tf.float32,
                                [None, FLAGS.patch_size, FLAGS.patch_size + FLAGS.disp_range - 1, num_channels],
                                name='rimage')
        targets = tf.placeholder(tf.float32, [None, FLAGS.disp_range], name='targets')

        snet = nf.create(limage, rimage, targets, state,FLAGS.net_type)
        prod = snet['inner_product']
        predicted = tf.argmax(prod, axis=1)
        acc_count = 0

        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(path))

        for i in range(0, lpatch.shape[0], FLAGS.eval_size):
            eval_dict = {limage: lpatch[i: i + FLAGS.eval_size],
                            rimage: rpatch[i: i + FLAGS.eval_size], snet['is_training']: False}
            pred = session.run([predicted], feed_dict=eval_dict)
            acc_count += np.sum(np.abs(pred - labels[i: i + FLAGS.eval_size]) <= 3)
            print('iter. %d finished, with %d correct (3-pixel error)' % (i + 1, acc_count))

            print('accuracy: %.3f' % ((acc_count / lpatch.shape[0]) * 100))
    #cuda.select_device(0)
    #cuda.close()
    tf.reset_default_graph()
    return ((acc_count / lpatch.shape[0]) * 100)
def Main(path):
    best=math.inf
    num=0
    while(best>0):
        state=TwoD_Generator()
        t_flops = train(state, num)
        # print('self num=',self.num)
        acc = evaluate(state, num)
        # acc=acc/100
        # flops=(100000000-t_flops)/100000000
        # e=0.5*(1-acc)+0.5*(1-flops)
        # e=1/(acc*t_flops)
        if acc == 0.0:
            e = math.inf
        else:
            e = t_flops / acc
        statea = str(state)
        print(colored('rggtrggegergggwgegwgw', 'yellow'), e)
        conn = sqlite3.connect(path)
        c = conn.cursor()
        c.execute('''INSERT INTO _all_ VALUES (?,?,?,?,?)''', [num, statea, acc, t_flops, e])
        conn.commit()
        conn.close()

        if e < best:
            conn = sqlite3.connect(path)
            c = conn.cursor()
            c.execute('''INSERT INTO bestss VALUES (?,?,?,?,?)''', [num, statea, acc, t_flops, e])
            conn.commit()
            conn.close()
            best = e
            print(colored('4ogmmregreomgerogmreomerormfrofmfoemwfoewmfewofm', 'red'))
        num = num + 1
        print(str(now))
        print(state)

    return e

if __name__=='__main__':
    path = FLAGS.model_dir + '\\bests.db'
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute('''CREATE TABLE bestss
                         (num int, arc text, acc real, t_flops real, energy real)''')
    conn.commit()
    c = conn.cursor()
    c.execute('''CREATE TABLE _all_
                                (num int, arc text, acc real, t_flops real, energy real)''')
    conn.commit()
    conn.close()
    Main(path)


