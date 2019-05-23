from IPython import display

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import os
from utils import Logger

import tensorflow as tf
from tensorflow import nn, layers
from tensorflow.contrib import layers as clayers 

import numpy as np

# Output Directory
ROOT_RESULT_PATH = os.path.abspath(__file__ + "../../../../")+'/output/dcgans/'
GENERATOR_SCOPE = "GAN/Generator"
DISCRIMINATOR_SCOPE = "GAN/Discriminator"

def cifar_data():
    compose = transforms.Compose([transforms.Resize(64),transforms.ToTensor(),transforms.Normalize((.5, .5, .5), (.5, .5, .5)),])
    return datasets.CIFAR10(root='dataset/CIFAR/', train=True, download=True, transform=compose)

dataset = cifar_data()
batch_size = 100
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_batches = len(dataloader)

IMAGES_SHAPE = (64, 64, 3)
NOISE_SIZE = 100

def default_conv2d(inputs, filters):
    return layers.conv2d(
        inputs,
        filters=filters,
        kernel_size=4,
        strides=(2, 2),
        padding='same',
        data_format='channels_last',
        use_bias=False,
    )

def default_conv2d_transpose(inputs, filters):
    return layers.conv2d_transpose(
        inputs,
        filters=filters,
        kernel_size=4,
        strides=(2, 2),
        padding='same',
        data_format='channels_last',
        use_bias=False,
    )

def noise(n_rows, n_cols):
    return np.random.normal(size=(n_rows, n_cols))

def discriminator(x):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        with tf.variable_scope("conv1"):
            conv1 = default_conv2d(x, 128)
            conv1 = nn.leaky_relu(conv1,alpha=0.2)
        
        with tf.variable_scope("conv2"):
            conv2 = default_conv2d(conv1, 256)
            conv2 = layers.batch_normalization(conv2)
            conv2 = nn.leaky_relu(conv2,alpha=0.2)
            
        with tf.variable_scope("conv3"):
            conv3 = default_conv2d(conv2, 512)
            conv3 = layers.batch_normalization(conv3)
            conv3 = nn.leaky_relu(conv3,alpha=0.2)
            
        with tf.variable_scope("conv4"):
            conv4 = default_conv2d(conv3, 1024)
            conv4 = layers.batch_normalization(conv3)
            conv4 = nn.leaky_relu(conv3,alpha=0.2)
        
        with tf.variable_scope("linear"):
            linear = clayers.flatten(conv4)
            linear = clayers.fully_connected(linear, 1)
        
        with tf.variable_scope("out"):
            out = nn.sigmoid(linear)
    return out

def generator(z):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
    
        with tf.variable_scope("linear"):
            linear = clayers.fully_connected(z, 1024 * 4 * 4)
            
        with tf.variable_scope("conv1_transp"):
            # Reshape as 4x4 images
            conv1 = tf.reshape(linear, (-1, 4, 4, 1024))
            conv1 = default_conv2d_transpose(conv1, 512)
            conv1 = layers.batch_normalization(conv1)
            conv1 = nn.relu(conv1)
        
        with tf.variable_scope("conv2_transp"):
            conv2 = default_conv2d_transpose(conv1, 256)
            conv2 = layers.batch_normalization(conv2)
            conv2 = nn.relu(conv2)
            
        with tf.variable_scope("conv3_transp"):
            conv3 = default_conv2d_transpose(conv2, 128)
            conv3 = layers.batch_normalization(conv3)
            conv3 = nn.relu(conv3)
            
        with tf.variable_scope("conv4_transp"):
            conv4 = default_conv2d_transpose(conv3, 3)
        
        with tf.variable_scope("out"):
            out = tf.tanh(conv4)
    return out

## Real Input
real_sample = tf.placeholder(tf.float32, shape=(None, )+IMAGES_SHAPE)
## Latent Variables / Noise
noise_sample = tf.placeholder(tf.float32, shape=(None, NOISE_SIZE))

# Generator
G_sample = generator(noise_sample)
# Discriminator
D_real = discriminator(real_sample)
D_fake = discriminator(G_sample)

# Generator
G_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake, labels=tf.ones_like(D_fake)
    )
)

# Discriminator
D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_real, labels=tf.ones_like(D_real)
    )
)

D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake, labels=tf.zeros_like(D_fake)
    )
)

D_loss = D_loss_real + D_loss_fake

# Obtain trainable variables for both networks
train_vars = tf.trainable_variables()

G_vars = [var for var in train_vars if 'generator' in var.name]
D_vars = [var for var in train_vars if 'discriminator' in var.name]

num_epochs = 200

G_opt = tf.train.AdamOptimizer(2e-4).minimize(G_loss, var_list=G_vars,)
D_opt = tf.train.AdamOptimizer(2e-4).minimize(D_loss, var_list=D_vars,)

num_test_samples = 16
test_noise = noise(num_test_samples, NOISE_SIZE)

BATCH_SIZE = 100
NUM_EPOCHS = 200

# session = tf.InteractiveSession()
# tf.global_variables_initializer().run(session=session)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
logger = Logger(model_name='DCGAN1', data_name='CIFAR10', root_path=ROOT_RESULT_PATH)

# Iterate through epochs
for epoch in range(NUM_EPOCHS):
    for n_batch, (batch,_) in enumerate(dataloader):
        
        # 1. Train Discriminator
        X_batch = batch.permute(0, 2, 3, 1).numpy()
        feed_dict = {real_sample: X_batch, noise_sample: noise(BATCH_SIZE, NOISE_SIZE)}
        _, d_error, d_pred_real, d_pred_fake = sess.run([D_opt, D_loss, D_real, D_fake], feed_dict=feed_dict)

        # 2. Train Generator
        feed_dict = {noise_sample: noise(BATCH_SIZE, NOISE_SIZE)}
        _, g_error = sess.run([G_opt, G_loss], feed_dict=feed_dict)

        # if n_batch % 10 == 0:
        logger.display_status(epoch, num_epochs, n_batch, num_batches,d_error, g_error, d_pred_real, d_pred_fake)
        
        if n_batch % 100 == 0:
            display.clear_output(True)
            # Generate images from test noise
            test_images = sess.run(G_sample, feed_dict={noise_sample: test_noise})
            # Log Images
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches, format='NHWC');
            # Log Status
            logger.display_status(epoch, num_epochs, n_batch, num_batches,d_error, g_error, d_pred_real, d_pred_fake)