import tensorflow as tf

slim = tf.contrib.slim


def create_network(state, inputs, is_training, scope="win37_dep9", reuse=False):
    num_maps = 64
    kw = 5
    kh = 5
    #print('state:',state)

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):

            '''net = slim.conv2d(inputs, 32, [kh, kw], scope='conv_bn_relu1')


            net = slim.conv2d(net, 32, [kh, kw], scope='conv_bn_relu2')
            net = slim.repeat(net, 6, slim.conv2d, num_maps, [kh, kw], scope='conv_bn_relu3_8')'''
            '''f=state.popleft()
            if f[0]=='conv2d':
                net=slim.conv2d(inputs,f[1],[kh,kw],padding=f[2],scope='0')
            else:
                net = slim.conv2d(inputs, f[1], [kh, kw],padding=f[2] ,scope='0')
                print('else')
            for i in state:
                if i[0] == 'conv2d':
                    net = slim.conv2d(net, i[1], [kh, kw],padding=i[2] ,scope=str(state.index(i)))
                else:
                    net = slim.conv2d(net, i[1], [kh, kw],padding=i[2], scope=str(state.index(i)))
                    print('else')'''
            '''tensors=[]
            number=0
            net = slim.conv2d(inputs, 32, [kh, kw], scope='conv_bn_relu1')
            tensors.append(net)
            for i in state:
                if i[0]=='conv2d':
                    net=slim.conv2d(net,state[1],[kh,kw],padding=state[2],scope=str(number))
                elif i[0]=='batch':
                    net=slim.batch_norm(net,is_training=is_training)
                else:pass





            net = slim.conv2d(net, num_maps, [kh, kw], scope='conv9', activation_fn=None, 
                    normalizer_fn=None)'''
            # print(type(state))
            '''for i in state:
                if i[0]=='none':
                    state.remove(i)
            #print(state)

            f=state.pop(0)
            if f[0]=='conv2d':
                net=slim.conv2d(inputs,64,[f[3],f[3]],padding=f[2],scope='conv1')
                #print(net.shape)
            elif f[0]=='batch':
                net=slim.batch_norm(inputs,is_training=is_training)
                #print(net.shape)
            else:
                #print(f)
                raise ValueError('not in category1')
            for i in state:
                if i[0] == 'conv2d':
                    net = slim.conv2d(net, i[1], [i[3], i[3]], padding=i[2])
                elif i[0] == 'batch':
                    net = slim.batch_norm(net, is_training=is_training)
                elif i[0]=='none':
                    pass
                else:
                    #print(i)
                    raise ValueError('not in category')
                #print(net.shape)

            net = slim.batch_norm(net, is_training=is_training)	'''

            #print('state in win37',state)
            '''net = [inputs] * len(state)
            print('net0',net)
            #print('gigntontgnerogerognewofg', len(net)) please
            indx = 0
            print(len(state))

            for i in state:
                # indx = state.index(i)
                #print('indx', indx)
                #print('len', len(state))
                #print('len1', len(net))
                print(i)

                # net[indx] = inputs
                for j in i:
                    if j[0] == 'conv2d':
                        net[indx] = slim.conv2d(net[indx], j[1], [j[3], j[3]], padding=j[2])
                        #print(j[0]) please
                    elif j[0] == 'batch':
                        net[indx] = slim.batch_norm(net[indx], is_training=is_training)
                        #print(j[0]) please
                    elif j[0] == 'conc':
                        shape = net[j[1]].shape
                        #print(shape) please
                        temp_net = tf.image.resize_images(images=net[indx], size=(shape[1], shape[2]))

                        temp_net = slim.conv2d(temp_net, shape[3], [5, 5], padding='same')
                        net[j[1]] += temp_net
                        #print(j[0]) please
                    else:
                        pass
                        #print(j[0]) please
                indx += 1'''
            '''for i in net:
                if i==None:
                    net.remove(i)'''
            #print(net) please

            '''for ind in net:
                ind = slim.batch_norm(ind, is_training=is_training)'''
            '''for i in range(len(net)):
                net[i] = slim.conv2d(net[i], 64, [5, 5], padding='same')
                net[i] = slim.batch_norm(net[i], is_training=is_training)'''
            has=False
            #print(state[1])
            for i in state[1]:
                if i[0]!='none':
                    has=True
            if has==False:
                print('effefegrghrg',state)
                net=inputs
                for i in state[0]:
                    if i[0] == 'conv2d':
                        print(i)
                        net = slim.conv2d(net, i[1], [i[3], i[3]], padding=i[2])
                        # print(net.shape)
                    elif i[0] == 'batch':
                        net = slim.batch_norm(net, is_training=is_training)
                        # print(net.shape)
                    else:
                        pass
                        # print(f)
                        #raise ValueError('not in category1')



                net = slim.batch_norm(net, is_training=is_training)
                return net


            else:
                nets=[inputs,inputs]
                f1=f2=['nome',64,'none',37]
                for i in range(max(len(state[0])+1,len(state[1])+1)):
                    if len(state[0])!=0:
                        f3=f1
                        f1=state[0].pop(0)
                        if f1[0] == 'conv2d':
                            nets[0] = slim.conv2d(nets[0], f1[1], [f1[3], f1[3]], padding=f1[2])
                            # print(net.shape)
                        elif f1[0] == 'batch':
                            nets[0] = slim.batch_norm(nets[0], is_training=is_training)
                            # print(net.shape)
                        elif f1[0]=='conc':
                            shape=nets[0].shape
                            print(shape)
                            print(shape[1],type(shape[1]))
                            x=int(shape[1])
                            y=int(shape[2])
                            print(x,type(x))
                            print(nets[1].shape)
                            #input()
                            new_image=tf.image.resize_image_with_crop_or_pad(nets[1],x,y)
                            print(new_image)
                            print(f3)
                            new_image=slim.conv2d(new_image,nets[0].shape[3],[3,3],padding='same')
                            new_image=slim.batch_norm(new_image,is_training=is_training)
                            print(nets[0].shape)
                            print(new_image.shape)
                            nets[0]=nets[0]+new_image
                        else:
                            pass
                    if len((state[1]))!=0:
                        f4=f2
                        f2 = state[1].pop(0)
                        if f2[0] == 'conv2d':
                            print('kkkkk',f2)
                            nets[1] = slim.conv2d(nets[1], f2[1], [f2[3], f2[3]], padding=f2[2])
                            # print(net.shape)
                        elif f2[0] == 'batch':
                            nets[1] = slim.batch_norm(nets[1], is_training=is_training)
                            # print(net.shape)
                        elif f2[0] == 'conc':
                            shape = nets[1].shape
                            print(shape)
                            x=int(shape[1])
                            y=int(shape[2])
                            #input()
                            new_image = tf.image.resize_image_with_crop_or_pad(nets[0],x,y)
                            new_image = slim.conv2d(new_image, nets[1].shape[3], [3,3], padding='same')
                            new_image = slim.batch_norm(new_image, is_training=is_training)
                            nets[1] = nets[1] + new_image
                        else:
                            print(nets[1].shape)

                print('0',nets[0].shape)
                print('1',nets[1].shape)
                print(f1)
                filt=int(nets[0].shape[3])
                nets[1]=slim.conv2d(nets[1],filt,[1,1],padding='same')
                nets[1]=slim.batch_norm(nets[1],is_training=is_training)
                net=nets[0]+nets[1]
                return net


    #print('ffsdfd', len(net))
    '''net1 = net.pop(0)
    #print('gerggeg', len(net))
    for i in net:
        net1 += i
    net1=slim.batch_norm(net1,is_training=False)

    return net1'''