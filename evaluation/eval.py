import numpy as np
import matplotlib.pyplot as plt
import glob
import caffe
from matplotlib import pyplot as plt

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

images_to_test = glob.glob('/home/localadm/ICNet/evaluation/samplelist/*.jpg')
for image_path in images_to_test:
    im = caffe.io.load_image(image_path)
    print im.shape
    im = caffe.io.resize_image(im, 
            (input_image_size_h,
             input_image_size_w),
             interp_order=3)
    im_mean = np.zeros((input_image_size_h, input_image_size_w, 3))
    im_mean[:,:,0] = mean_image_r
    im_mean[:,:,1] = mean_image_g
    im_mean[:,:,2] = mean_image_b
    print im
    im = (im*255 - im_mean)/255

    print im.shape
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    out = net.forward()
    seg_result = out['conv6_interp']
    seg_result = seg_result[0,:,:,:]
    aux_layer_cls = seg_result
    print aux_layer_cls.shape
    aux_layer_cls = np.exp(aux_layer_cls)
    aux_layer_cls = aux_layer_cls/(np.add(aux_layer_cls, 3))
    aux_layer_cls = aux_layer_cls.argmax(0)
    fig = plt.figure("Segmentation")
    plt.imshow(aux_layer_cls, interpolation='nearest')
    plt.show()