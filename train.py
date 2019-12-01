import os
import tensorflow as tf
import models.net_factory as nf
import numpy as np
from data_handler import Data_handler
from simanneal import Annealer
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
flags.DEFINE_string('model_dir', 'C:\\Users\\Mohammad\\PycharmProjects\\version6\\ded', 'Trained network dir')
flags.DEFINE_string('data_version', 'kitti2015', 'kitti2012 or kitti2015')
flags.DEFINE_string('data_root', 'C:\\Users\\Mohammad\\Downloads\\data_scene_flow\\training', 'training dataset dir')
flags.DEFINE_string('util_root', 'C:\\Users\\Mohammad\\Downloads\\data_scene_flow', 'Binary training files dir')
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


class SimulatedAnnealer(Annealer):
    def __init__(self, state):
        self.num = 0
        self.best=math.inf
        self.path=FLAGS.model_dir+'\\bests.db'
        conn=sqlite3.connect(self.path)
        c=conn.cursor()
        c.execute('''CREATE TABLE bestss
                     (num int, arc text, acc real, t_flops real, energy real)''')
        conn.commit()
        c = conn.cursor()
        c.execute('''CREATE TABLE _all_
                            (num int, arc text, acc real, t_flops real, energy real)''')
        conn.commit()
        conn.close()
        super(SimulatedAnnealer, self).__init__(state)

    def move(self):

        #print('////////////////////////////////////////////')
        #print('first len==',len(self.state))
        #print(1)
        '''valids=[]
        #print(2)
        others=[]
        #print(3)
        layers=[['conv2d',32,'same',3],['conv2d',32,'same',5],['conv2d',32,'same',7],['conv2d',32,'same',11],['batch',0,'none',0],
                ['none',0,'none',0]]
        #print(4)
        x=random.choice(self.state)
        #j=self.state.index(x)
        for z in self.state:
            if z!=x:
                layers.append(['conc',self.state.index(z),None,0])
                #print('sj=ewfj')
        #print('rgormogmgmgoemegegrgregegregerkgntrntrgngoeg')
        #print(layers)
        #print('fergttrmtrgrmgormgtrogmogmerogmeogmegomegoegere')
        for i in x:
            #print(5)
            if i[0]=='conv2d':
                #print(6)
                if i[2]=='valid':
                    #print(7)
                    #print('vvvvvvvvvvvvvvvvvvv',i[3])
                    if i[3]>3:
                        #print(8)


                        valids.append(x.index(i))
                        #print(9)
                    else:
                        #print('dw')
                        pass
                else:
                    #print(10)
                    others.append(x.index(i))
                    #print(11)
            else:
                #print(12)
                others.append(x.index(i))
                #print(13)
        if len(valids)!=0:
            #print(14)

            if random.random()<0.2:
                #print(15)
                #print('valid change')

                #print('valids=',valids)
                a=random.choice(valids)
                print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
                for i in valids:
                    print(self.state[i])
                print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
                #print(16)
                #print('state index',a)
                #print(self.state[a])
                b=random.randrange(3,x[a][3],2)
                #print(17)
                #print('kernel',b)

                #print('hrt',b)
                temp=x[a][3]
                #print(18)
                #print('temp',temp)
                x[a][3]=b
                #print(19)
                #print('starteter',self.state[a])
                x.append(['conv2d',32,'valid',(temp-b)+1])
                #print(20)
                print(self.state)'''

        '''elif random.random()>=0.2 and random.random()<0.7:
                #print(21)
                print('others change')
                print('ergthyjukyjtrewq ertyujhtrew')
                for i in others:
                    print(self.state[i])
                print('fwrgthrgegregggergrgrgergegge')
                #print(22)
                a=random.choice(others)
                #print(23)
                be=random.choice(layers)
                #print(24)
                #print(a)
                #print(25)
                #print('be',be)
                #print(26)
                #self.state[a]=be
                x.remove(x[a])
                x.append(be)
                #print(27)
            else:
                #print(28)
                print('sweap')
                #print(29)

                a=random.randint(0,len(x)-1)
                #print(30)
                b=random.randint(0,len(x)-1)
                #print(a,b)
                #print(31)
                temp1=x[a]
                x[a]=x[b]
                x[b]=temp1
                #self.state[a],self.state[b]=self.state[b],self.state[a]
                #print(32)
        else:
            #print(33)
            print('else')
            #print(34)
            if random.random()<0.5:
                #print(35)
                print('others change')
                #print(36)
                a = random.choice(others)
                #print(37)
                #print(a)
                #print(38)
                be = random.choice(layers)
                #print(39)
                #print('be', be)
                #print(40)
                #self.state[a] = be
                x.remove(x[a])
                x.append(be)
                #print(41)
            else:
                #print(42)
                print('sweap')
                #print(43)
                a = random.randint(0, len(x) - 1)
                #print(44)
                b = random.randint(0, len(x) - 1)
                #print(45)
                temp1 = x[a]
                x[a] = x[b]
                x[b] = temp1
                #self.state[a], self.state[b] = self.state[b], self.state[a]
                #print(46)
        #print(len(x),x)
        kernel_sum=0
        num_node=0
        for i in x:
            if i[0]=='conv2d':
                if i[2]=='valid':
                    kernel_sum+=i[3]
                    num_node+=1
        ex=kernel_sum-num_node
        if ex!=36:
            #print('2',self.state)
            x.append(['conv2d',32,'valid',(36-ex)+1])
            #raise ValueError('this is not appropriante')
        #print(47)
        #print('self state type',type(x))'''
        selection=random.choice([True,False])
        if selection:
            valids = []
            # print(2)
            others = []
            # print(3)
            layers = [['conv2d', 32, 'same', 3], ['conv2d', 32, 'same', 5], ['conv2d', 32, 'same', 7],
                      ['conv2d', 32, 'same', 11], ['batch', 0, 'none', 0],['conc',0,'none',0],
                      ['none', 0, 'none', 0]]
            # print(4)
            x = self.state[0]
            # j=self.state.index(x)
            '''for z in self.state:
                if z != x:
                    layers.append(['conc', self.state.index(z), None, 0])
                    # print('sj=ewfj')'''
            # print('rgormogmgmgoemegegrgregegregerkgntrntrgngoeg')
            # print(layers)
            # print('fergttrmtrgrmgormgtrogmogmerogmeogmegomegoegere')
            for i in x:
                # print(5)
                if i[0] == 'conv2d':
                    # print(6)
                    if i[2] == 'valid':
                        # print(7)
                        # print('vvvvvvvvvvvvvvvvvvv',i[3])
                        if i[3] > 3:
                            # print(8)

                            valids.append(x.index(i))
                            # print(9)
                        else:
                            # print('dw')
                            pass
                    else:
                        # print(10)
                        others.append(x.index(i))
                        # print(11)
                else:
                    # print(12)
                    others.append(x.index(i))
                    # print(13)
            if len(valids) != 0:
                # print(14)

                if random.random() < 0.2:
                    # print(15)
                    # print('valid change')

                    # print('valids=',valids)
                    a = random.choice(valids)
                    '''print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
                    for i in valids:
                        print(self.state[i])
                    print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')'''
                    # print(16)
                    # print('state index',a)
                    # print(self.state[a])
                    b = random.randrange(3, x[a][3], 2)
                    # print(17)
                    # print('kernel',b)

                    # print('hrt',b)
                    temp = x[a][3]
                    # print(18)
                    # print('temp',temp)
                    x[a][3] = b
                    # print(19)
                    # print('starteter',self.state[a])
                    x.append(['conv2d', 32, 'valid', (temp - b) + 1])
                    # print(20)
                    print(self.state)


                elif random.random() >= 0.2 and random.random() < 0.7:
                    # print(21)
                    print('others change')
                    '''print('ergthyjukyjtrewq ertyujhtrew')
                    for i in others:
                        print(self.state[i])
                    print('fwrgthrgegregggergrgrgergegge')'''
                    # print(22)
                    a = random.choice(others)
                    # print(23)
                    be = random.choice(layers)
                    # print(24)
                    # print(a)
                    # print(25)
                    # print('be',be)
                    # print(26)
                    # self.state[a]=be
                    x.remove(x[a])
                    x.append(be)
                    # print(27)
                else:
                    # print(28)
                    print('sweap')
                    # print(29)

                    a = random.randint(0, len(x) - 1)
                    # print(30)
                    b = random.randint(0, len(x) - 1)
                    # print(a,b)
                    # print(31)
                    temp1 = x[a]
                    x[a] = x[b]
                    x[b] = temp1
                    # self.state[a],self.state[b]=self.state[b],self.state[a]
                    # print(32)
            else:
                # print(33)
                print('else')
                # print(34)
                if random.random() < 0.5:
                    # print(35)
                    print('others change')
                    # print(36)
                    a = random.choice(others)
                    # print(37)
                    # print(a)
                    # print(38)
                    be = random.choice(layers)
                    # print(39)
                    # print('be', be)
                    # print(40)
                    # self.state[a] = be
                    x.remove(x[a])
                    x.append(be)
                    # print(41)
                else:
                    # print(42)
                    print('sweap')
                    # print(43)
                    a = random.randint(0, len(x) - 1)
                    # print(44)
                    b = random.randint(0, len(x) - 1)
                    # print(45)
                    temp1 = x[a]
                    x[a] = x[b]
                    x[b] = temp1
                    # self.state[a], self.state[b] = self.state[b], self.state[a]
                    # print(46)
            # print(len(x),x)
            kernel_sum = 0
            num_node = 0
            for i in x:
                if i[0] == 'conv2d':
                    if i[2] == 'valid':
                        kernel_sum += i[3]
                        num_node += 1
            ex = kernel_sum - num_node
            if ex != 36:
                # print('2',self.state)
                x.append(['conv2d', 32, 'valid', (36 - ex) + 1])
                # raise ValueError('this is not appropriante')
            # print(47)
            # print('self state type',type(x))
        else:
            initialized=False
            for i in self.state[1]:
                if i[0]!='none':
                    initialized=True
            if initialized==False:
                self.state[1][len(self.state[1])-1]=['conv2d',32,'valid',37]
            else:
                valids = []
                # print(2)
                others = []
                # print(3)
                layers = [['conv2d', 32, 'same', 3], ['conv2d', 32, 'same', 5], ['conv2d', 32, 'same', 7],
                          ['conv2d', 32, 'same', 11], ['batch', 0, 'none', 0],['conc',0,'none',0],
                          ['none', 0, 'none', 0]]
                # print(4)
                x = self.state[1]
                # j=self.state.index(x)
                '''for z in self.state:
                    if z != x:
                        layers.append(['conc', self.state.index(z), None, 0])
                        # print('sj=ewfj')'''
                # print('rgormogmgmgoemegegrgregegregerkgntrntrgngoeg')
                # print(layers)
                # print('fergttrmtrgrmgormgtrogmogmerogmeogmegomegoegere')
                for i in x:
                    # print(5)
                    if i[0] == 'conv2d':
                        # print(6)
                        if i[2] == 'valid':
                            # print(7)
                            # print('vvvvvvvvvvvvvvvvvvv',i[3])
                            if i[3] > 3:
                                # print(8)

                                valids.append(x.index(i))
                                # print(9)
                            else:
                                # print('dw')
                                pass
                        else:
                            # print(10)
                            others.append(x.index(i))
                            # print(11)
                    else:
                        # print(12)
                        others.append(x.index(i))
                        # print(13)
                if len(valids) != 0:
                    # print(14)

                    if random.random() < 0.2:
                        # print(15)
                        # print('valid change')

                        # print('valids=',valids)
                        a = random.choice(valids)
                        # print(16)
                        # print('state index',a)
                        # print(self.state[a])
                        b = random.randrange(3, x[a][3], 2)
                        # print(17)
                        # print('kernel',b)

                        # print('hrt',b)
                        temp = x[a][3]
                        # print(18)
                        # print('temp',temp)
                        x[a][3] = b
                        # print(19)
                        # print('starteter',self.state[a])
                        x.append(['conv2d', 32, 'valid', (temp - b) + 1])
                        # print(20)
                        print(self.state)


                    elif random.random() >= 0.2 and random.random() < 0.7:
                        # print(21)
                        print('others change')
                        '''print('ergthyjukyjtrewq ertyujhtrew')
                        for i in others:
                            print(self.state[i])
                        print('fwrgthrgegregggergrgrgergegge')'''
                        # print(22)
                        a = random.choice(others)
                        # print(23)
                        be = random.choice(layers)
                        # print(24)
                        # print(a)
                        # print(25)
                        # print('be',be)
                        # print(26)
                        # self.state[a]=be
                        x.remove(x[a])
                        x.append(be)
                        # print(27)
                    else:
                        # print(28)
                        print('sweap')
                        # print(29)

                        a = random.randint(0, len(x) - 1)
                        # print(30)
                        b = random.randint(0, len(x) - 1)
                        # print(a,b)
                        # print(31)
                        temp1 = x[a]
                        x[a] = x[b]
                        x[b] = temp1
                        # self.state[a],self.state[b]=self.state[b],self.state[a]
                        # print(32)
                else:
                    # print(33)
                    print('else')
                    # print(34)
                    if random.random() < 0.5:
                        # print(35)
                        print('others change')
                        # print(36)
                        a = random.choice(others)
                        # print(37)
                        # print(a)
                        # print(38)
                        be = random.choice(layers)
                        # print(39)
                        # print('be', be)
                        # print(40)
                        # self.state[a] = be
                        x.remove(x[a])
                        x.append(be)
                        # print(41)
                    else:
                        # print(42)
                        print('sweap')
                        # print(43)
                        a = random.randint(0, len(x) - 1)
                        # print(44)
                        b = random.randint(0, len(x) - 1)
                        # print(45)
                        temp1 = x[a]
                        x[a] = x[b]
                        x[b] = temp1
                        # self.state[a], self.state[b] = self.state[b], self.state[a]
                        # print(46)
                # print(len(x),x)
                kernel_sum = 0
                num_node = 0
                for i in x:
                    if i[0] == 'conv2d':
                        if i[2] == 'valid':
                            kernel_sum += i[3]
                            num_node += 1
                ex = kernel_sum - num_node
                if ex != 36:
                    # print('2',self.state)
                    x.append(['conv2d', 32, 'valid', (36 - ex) + 1])
                    # raise ValueError('this is not appropriante')
                # print(47)
                # print('self state type',type(x))

        return self.energy()


    def energy(self):
        #print('state in energy function',self.state)
        '''for i in self.state:
            #print(self.state)
            #print('remove none procedure')
            #print(i)
            isit=False
            for x in i:
                if x[0]!='none':
                    isit=True
                    #print('           -------            ')
            #print(isit)
            if isit==False:
                #print('why')
                self.state.pop(self.state.index(i))'''
        #print('ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff')
        #print(len(self.state))
        #print('>>>',self.state)
        has=False
        for i in self.state[1]:
            if i[0]!='none':
                has=True


        if has==True:
            for x in self.state:
                print('x:',x)
                kernel_sum = 0
                num_node = 0
                for i in x:
                    if i[0]=='conv2d':
                        if i[2]=='valid':
                            kernel_sum+=i[3]
                            num_node+=1
                ex=kernel_sum-num_node
                if ex!=36:
                    print('2',x)
                    raise ValueError('this is not appropriante')
        else:
            #print('x:', x)
            kernel_sum = 0
            num_node = 0
            for i in self.state[0]:
                if i[0] == 'conv2d':
                    if i[2] == 'valid':
                        kernel_sum += i[3]
                        num_node += 1
            ex = kernel_sum - num_node
            if ex != 36:
                #print('2', x)
                raise ValueError('this is not appropriante')




        #print('self num=',self.num)
        t_flops=train(self.state, self.num)
        #print('self num=',self.num)
        acc = evaluate(self.state,self.num)
        #acc=acc/100
        #flops=(100000000-t_flops)/100000000
        #e=0.5*(1-acc)+0.5*(1-flops)
        #e=1/(acc*t_flops)
        if acc==0.0:
            e=math.inf
        else:
            e=t_flops/acc
        statea=str(self.state)
        print(colored('rggtrggegergggwgegwgw','yellow'),e)
        conn = sqlite3.connect(self.path)
        c = conn.cursor()
        c.execute('''INSERT INTO _all_ VALUES (?,?,?,?,?)''', [self.num, statea, acc, t_flops, e])
        conn.commit()
        conn.close()

        if e<self.best:
            conn = sqlite3.connect(self.path)
            c = conn.cursor()
            c.execute('''INSERT INTO bestss VALUES (?,?,?,?,?)''',[self.num,statea,acc,t_flops,e])
            conn.commit()
            conn.close()
            self.best=e
            print(colored('4ogmmregreomgerogmreomerormfrofmfoemwfoewmfewofm','red'))
        self.num = self.num + 1
        print(str(now))
        print(self.state)

        return e
        #return random.random()


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


'''if FLAGS.phase == 'train':
	train()
elif FLAGS.phase == 'evaluate': 
	evaluate()
else:
	sys.exit('FLAGS.phase = train or evaluate')'''
''' [['conv2d', 32, 'same', 5],
             ['conv2d', 64, 'same', 5],
             ['conc', 0, 'none', 0],
             ['conv2d', 64, 'same', 5],
             ['none', 0, 'none', 0],
             ['conv2d', 64, 'same', 5],
             ['conv2d', 64, 'same', 5],
             ['none', 0, 'none', 0],
             ['conv2d', 64, 'same', 5],
             ['conv2d', 64, 'same', 5],
             ['conv2d', 64, 'valid', 37]]'''
if __name__ == '__main__':
    '''init = [[['conv2d', 32, 'same', 5],
             ['conv2d', 64, 'same', 5],
             ['none', 0, 'none', 0],
             ['conv2d', 64, 'same', 5],
             ['none', 0, 'none', 0],
             ['conv2d', 64, 'same', 5],
             ['conv2d', 64, 'same', 5],
             ['none', 0, 'none', 0],
             ['conv2d', 64, 'same', 5],
             ['conv2d', 64, 'same', 5],
             ['conv2d', 64, 'valid', 37]]
            ,
            [['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0],
             ['none', 0, 'none', 0]]
            ]'''
    '''init=[[['conv2d', 32, 'same', 5],
           ['conv2d', 64, 'same', 5],
           ['none', 0, 'none', 0],
           ['conv2d', 64, 'same', 5],
           ['none', 0, 'none', 0],
           ['conv2d', 64, 'same', 5],
           ['conv2d', 64, 'same', 5],
           ['none', 0, 'none', 0],
           ['conv2d', 64, 'valid', 3],
           ['conv2d', 64, 'same', 5],
           ['conv2d', 64, 'same', 5],
           ['conv2d', 32, 'valid', 35]],
          [['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['none', 0, 'none', 0],
           ['conv2d', 32, 'valid', 37]]]'''
    init=[[['batch', 0, 'none', 0], ['conv2d', 32, 'valid', 13], ['none', 0, 'none', 0], ['conv2d', 32, 'same', 5], ['conv2d', 64, 'same', 5], ['conv2d', 64, 'valid', 3], ['conv2d', 64, 'same', 5], ['none', 0, 'none', 0], ['conv2d', 32, 'valid', 3], ['none', 0, 'none', 0], ['conv2d', 32, 'valid', 3], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['conv2d', 32, 'valid', 7], ['batch', 0, 'none', 0], ['conv2d', 32, 'valid', 3], ['conv2d', 32, 'valid', 11]], [['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['none', 0, 'none', 0], ['conv2d', 32, 'valid', 37]]]
    tsp = SimulatedAnnealer(init)
    #print('///////////////////////////////////////////////////////////////')
    tsp.set_schedule(tsp.auto(0.01,10))
    tsp.copy_strategy = "slice"
    state, e = tsp.anneal()
    print()
    print("%i mile rout:" % e)
    print("state {}".format(state))