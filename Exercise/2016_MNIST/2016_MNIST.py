import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import tensorflow_datasets as tfds

# See all registered datasets
# tfds.list_builders()

# Load a given dataset by name, along with the DatasetInfo
# data, info = tfds.load("nsynth", with_info=True)
# train_data, test_data = data['train'], data['test']
# assert isinstance(train_data, tf.data.Dataset)
# assert info.features['label'].num_classes == 10
# assert info.splits['train'].num_examples == 60000

# # You can also access a builder directly
# builder = tfds.builder("nsynth")
# assert builder.info.splits['train'].num_examples == 60000
# builder.download_and_prepare()
# datasets = builder.as_dataset()

# If you need NumPy arrays
# np_datasets = tfds.as_numpy(datasets)

# Output Directory
OUTPUT_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.output/')
DATASET_PATH = os.path.join(os.path.abspath(__file__+ "../../"), '.dataset/mnist/')
if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)
if not os.path.exists(DATASET_PATH): os.makedirs(DATASET_PATH)

mnist = input_data.read_data_sets(DATASET_PATH, one_hot=True)
X = tf.placeholder(tf.float32, [None, 28 * 28]) # MNIST = 28*28
Z = tf.placeholder(tf.float32, [None, 128]) # Noise Dimension = 128

# ********* G-Network (Hidden Node # = 256)
G_W1 = tf.Variable(tf.random_normal([128, 256], stddev=0.01))
G_W2 = tf.Variable(tf.random_normal([256, 28 * 28], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([256]))
G_b2 = tf.Variable(tf.zeros([28 * 28]))

def generator(noise_z): # 128 -> 256 -> 28*28
    hidden = tf.nn.leaky_relu(tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.tanh(tf.matmul(hidden, G_W2) + G_b2)
    return output

# ********* D-Network (Hidden Node # = 256)
D_W1 = tf.Variable(tf.random_normal([28 * 28, 256], stddev=0.01))
D_W2 = tf.Variable(tf.random_normal([256, 1], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([256]))
D_b2 = tf.Variable(tf.zeros([1]))

def discriminator(inputs): # 28*28 -> 256 -> 1
    hidden = tf.nn.leaky_relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)
    return output

# ********* Generation, Loss, Optimization and Session Init.
G = generator(Z)
loss_D = -tf.reduce_mean(tf.log(discriminator(X)) + tf.log(1 - discriminator(G)))
loss_G = -tf.reduce_mean(tf.log(discriminator(G)))
train_D = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss_D, var_list=[D_W1, D_b1, D_W2, D_b2])
train_G = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss_G, var_list=[G_W1, G_b1, G_W2, G_b2])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# ********* Training and Testing
noise_test = np.random.normal(size=(10, 128)) # 10 = Test Sample Size, 128 = Noise Dimension
for epoch in range(200): # 200 = Num. of Epoch
    for i in range(int(mnist.train.num_examples / 100)): # 100 = Batch Size
        batch_xs, _ = mnist.train.next_batch(100)
        noise = np.random.normal(size=(100, 128))

        sess.run(train_D, feed_dict={X: batch_xs, Z: noise})
        sess.run(train_G, feed_dict={Z: noise})

    if epoch == 0 or (epoch + 1) % 10 == 0: # 10 = Saving Period
        samples = sess.run(G, feed_dict={Z: noise_test})

        fig, ax = plt.subplots(1, 10, figsize=(10, 1))
        for i in range(10):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))
        plt.savefig(OUTPUT_PATH+'{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)