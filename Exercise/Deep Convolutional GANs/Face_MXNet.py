from __future__ import print_function
import os
import matplotlib as mpl
import tarfile
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
from mxnet import autograd
import numpy as np

print("######################################")
print("### Start DCGANs~~!!")
print("######################################")

# Set training parameter
EPOCHS = 2  # Set low by default for tests, set higher when you actually run this code.
BATCH_SIZE = 64
LATENT_Z_SIZE = 100

USE_GPU = False
CTX = mx.gpu() if USE_GPU else mx.cpu()

LR = 0.0002
BETA1 = 0.5

# Download and preprocess the LWF Face Dataset.
URL_PATH = 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
DATA_PATH = os.path.dirname( os.path.abspath( __file__ ))+'/lfw_dataset/'
FILE_NAME = 'lfw-deepfunneled.tgz'
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    data_file = utils.download(URL_PATH, DATA_PATH)
    print("FileName: " + data_file);
    with tarfile.open(data_file) as tar:
        tar.extractall

# Fist, we resize images to size 64 X 64. Then, we normalize all pixel values to the [-1,1] range.
RESIZE_WIDTH = 64
RESIZE_HEIGHT = 64
IMAGE_LIST = []

def transform(data, resize_width, resize_height):
    # resize to resize_width * resize_height
    data = mx.image.imresize(data, resize_width, resize_height)
    # transpose from (resize_width, resize_height, 3)
    # to (3, resize_width, target_ht)
    data = nd.transpose(data, (2, 0, 1))
    # normalize to [-1, 1]
    data = data.astype(np.float32) / 127.5 - 1
    # if image is greyscale, repeat 3 times to get RGB image.
    if data.shape[0] == 1:
        data = nd.tile(data, (3, 1, 1))
    return data.reshape((1,) + data.shape)


for path, _, fileNames in os.walk(DATA_PATH):
    for file in fileNames:
        if not file.endswith('.jpg'):
            continue
        print("FileName: " + file)
        imgPath = os.path.join(path, file)
        img_data = mx.image.imread(imgPath)
        img_data = transform(img_data, RESIZE_WIDTH, RESIZE_HEIGHT)
        IMAGE_LIST.append(img_data)

TRAIN_DATA = mx.io.NDArrayIter(data=nd.concatenate(IMAGE_LIST), batch_size=BATCH_SIZE)


# Visualize 4 images:
def visualize(img_arr):
    plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        visualize(IMAGE_LIST[i + 10][0])
    plt.show()


# [ Defining the networks ]
# build the generator
nc = 3
ngf = 64
netG = nn.Sequential()
with netG.name_scope():
    # input is Z, going into a convolution
    netG.add(nn.Conv2DTranspose(ngf * 8, 4, 1, 0, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 4 x 4
    netG.add(nn.Conv2DTranspose(ngf * 4, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 8 x 8
    netG.add(nn.Conv2DTranspose(ngf * 2, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 16 x 16
    netG.add(nn.Conv2DTranspose(ngf, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 32 x 32
    netG.add(nn.Conv2DTranspose(nc, 4, 2, 1, use_bias=False))
    netG.add(nn.Activation('tanh'))
    # state size. (nc) x 64 x 64
# build the discriminator
ndf = 64
netD = nn.Sequential()
with netD.name_scope():
    # input is (nc) x 64 x 64
    netD.add(nn.Conv2D(ndf, 4, 2, 1, use_bias=False))
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 32 x 32
    netD.add(nn.Conv2D(ndf * 2, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 16 x 16
    netD.add(nn.Conv2D(ndf * 4, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 8 x 8
    netD.add(nn.Conv2D(ndf * 8, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 4 x 4
    netD.add(nn.Conv2D(1, 4, 1, 0, use_bias=False))

# [ Setup Loss Function and Optimizer ]
# loss
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
# initialize the generator and the discriminator
netG.initialize(mx.init.Normal(0.02), ctx=CTX)
netD.initialize(mx.init.Normal(0.02), ctx=CTX)
# trainer for the generator and the discriminator
trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': LR, 'beta1': BETA1})
trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': LR, 'beta1': BETA1})

# [ Training Loop ]
from datetime import datetime
import time
import logging

real_label = nd.ones((BATCH_SIZE,), ctx=CTX)
fake_label = nd.zeros((BATCH_SIZE,), ctx=CTX)


def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


metric = mx.metric.CustomMetric(facc)

stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
logging.basicConfig(level=logging.DEBUG)

for epoch in range(EPOCHS):
    tic = time.time()
    btic = time.time()
    TRAIN_DATA.reset()
    iter = 0
    for batch in TRAIN_DATA:
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        data = batch.data[0].as_in_context(CTX)
        latent_z = mx.nd.random_normal(0, 1, shape=(BATCH_SIZE, LATENT_Z_SIZE, 1, 1), ctx=CTX)

        with autograd.record():
            # train with real image
            output = netD(data).reshape((-1, 1))
            errD_real = loss(output, real_label)
            metric.update([real_label, ], [output, ])

            # train with fake image
            fake = netG(latent_z)
            output = netD(fake.detach()).reshape((-1, 1))
            errD_fake = loss(output, fake_label)
            errD = errD_real + errD_fake
            errD.backward()
            metric.update([fake_label, ], [output, ])

        trainerD.step(batch.data[0].shape[0])

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        with autograd.record():
            fake = netG(latent_z)
            output = netD(fake).reshape((-1, 1))
            errG = loss(output, real_label)
            errG.backward()

        trainerG.step(batch.data[0].shape[0])

        # Print log infomation every ten batches
        if iter % 10 == 0:
            name, acc = metric.get()
            logging.info('speed: {} samples/s'.format(BATCH_SIZE / (time.time() - btic)))
            logging.info('discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
                         % (nd.mean(errD).asscalar(),
                            nd.mean(errG).asscalar(), acc, iter, epoch))
        iter = iter + 1
        btic = time.time()

    name, acc = metric.get()
    metric.reset()
    # logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
    # logging.info('time: %f' % (time.time() - tic))

    # Visualize one generated image for each epoch
    # fake_img = fake[0]
    # visualize(fake_img)
    # plt.show()

# [ Results ]
# Given a trained generator, we can generate some images of faces.
num_image = 8
for i in range(num_image):
    latent_z = mx.nd.random_normal(0, 1, shape=(1, LATENT_Z_SIZE, 1, 1), ctx=CTX)
    imgPath = netG(latent_z)
    plt.subplot(2, 4, i + 1)
    visualize(imgPath[0])
plt.show()

num_image = 12
latent_z = mx.nd.random_normal(0, 1, shape=(1, LATENT_Z_SIZE, 1, 1), ctx=CTX)
step = 0.05
for i in range(num_image):
    imgPath = netG(latent_z)
    plt.subplot(3, 4, i + 1)
    visualize(imgPath[0])
    latent_z += 0.05
plt.show()
