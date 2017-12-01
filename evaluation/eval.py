import numpy as np
import matplotlib.pyplot as plt
import glob
import caffe
from matplotlib import pyplot as plt
import cv2
from timeit import default_timer as timer

# EJECUTAR ESTO ANTES, POR SI ACASO
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/localadm/ICNet/PSPNet/build/lib/ && export PYTHONPATH=/home/localadm/ICNet/PSPNet/python:/home/localadm/ICNet/PSPNet/src/caffe/proto/

model_weights = 'model/icnet_cityscapes_train_30k.caffemodel'
model_deploy = 'prototxt/icnet_cityscapes.prototxt'
phase = 'test'

save_root = ''

input_image_size_h = 1025
input_image_size_w = 2049
mean_image_r = 123.68
mean_image_g = 116.779
mean_image_b = 103.939

gpu_id = 1

caffe.set_mode_gpu()
caffe.set_device(gpu_id)
net = caffe.Net(model_deploy, model_weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
transformer.set_mean('data',np.asarray([123.68, 116.779, 103.939]))

images_to_test = sorted(glob.glob('/media/localadm/data/dataset/20170530-12-31/front/*'))
for image_path in images_to_test:
    start = timer()
    im = caffe.io.load_image(image_path)
    im = caffe.io.resize_image(im, 
            (input_image_size_h,
             input_image_size_w),
             interp_order=3)

    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    prepross_t = timer()
    print('Preprocess: {}'.format(prepross_t-start))
    out = net.forward()
    forward_t = timer()
    print('forward: {}'.format(forward_t - prepross_t))
    aux_layer_cls = cv2.exp(out['conv6_interp'][0,:,:,:])
    aux_layer_cls = cv2.divide(aux_layer_cls, cv2.add(aux_layer_cls, 3))
    aux_layer_cls = aux_layer_cls.argmax(0)
    aux_layer_cls = np.divide(aux_layer_cls, 13.)
    cv2.imshow('frame', aux_layer_cls)
    end = timer()
    print('Postprocess: {}'.format(end - forward_t))
    print('Total time: {}\n'.format(end-start))
    cv2.waitKey(30)