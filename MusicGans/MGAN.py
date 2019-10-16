import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.io.wavfile
import librosa
import pywt
import tensorflow as tf
from TransSound import *

ts = TransSound()

#### DWT(Discrete Wavelet Transform) Samapling
samplingRate=22050 # 22.050kHz
samplingRate, realList, realData = ts.getSource(targetFile=ts.getSourceFile(), targeWavelet='db2', targetLevel=3);
cA3, cD3, cD2, cD1 = realList
print("< Discrete Wavelet Transform >\n" + "  cD1: {0}\n  cD2: {1}\n  cD3: {2}\n  cA3: {3}\n".format(cD1,cD2,cD3,cA3))


def createFakeData(source):
    print(str(source.max()) +","+str(source.min()))
    return (source.max()-source.min())*np.random.random(source.size)-(source.min()*-1)

# Fake Data
# fake_cD1 = (np.random.random(cD1.size)-0.5)*2 # [n x 1], -1.0 ~ +1.0
# fake_cD2 = (np.random.random(cD2.size)-0.5)*2
# fake_cD3 = (np.random.random(cD3.size)-0.5)*2
# fake_cA3 = (np.random.random(cA3.size)-0.5)*2
# fake_cD1 = cD1
# fake_cD2 = cD2
# fake_cD3 = cD3
# fake_cA3 = cA3
fake_cD1 = createFakeData(cD1);
fake_cD2 = createFakeData(cD2);
fake_cD3 = createFakeData(cD3);
fake_cA3 = createFakeData(cA3);
# 5 * np.random.random_sample((3, 2)) - 5 >> [-5 ~ 0]

def createMatrix(dA3, dD3, dD2, dD1):
    data = []
    tA3=tD3=tD2=tD1=0.0
    for i in range(dD1.size):
        if i < dA3.size: tA3 = dA3[i]
        if i < dD3.size: tD3 = dD3[i]
        if i < dD2.size: tD2 = dD2[i]
        tD1 = dD1[i]
        data.append([tA3, tD3, tD2, tD1])
    return np.array(data, dtype=dD1.dtype)

GENERATOR_SCOPE = "GAN/Generator"
DISCRIMINATOR_SCOPE = "GAN/Discriminator"

### Creates a fully connected neural network of 2 hidden layers
def createGeneratorNetowk(noise_placeholder, hsize=[16, 16], reuse=False): # Z: [none, 2]
    with tf.variable_scope(GENERATOR_SCOPE,reuse=reuse):
        # dense(inputs, units, activation ...)
        hidden1 = tf.layers.dense(noise_placeholder, hsize[0], activation=tf.nn.leaky_relu) # hidden1 Tensor name: GAN/Generator/dense/LeakyRelu:0, shape=(?, 16), dtype=float32
        hidden2 = tf.layers.dense(hidden1, hsize[1], activation=tf.nn.leaky_relu) # hidden2 name: GAN/Generator/dense_1/LeakyRelu:0, shape=(?, 16), dtype=float32
        out = tf.layers.dense(hidden2, 4) # out name: GAN/Generator/dense_2/BiasAdd:0, shape=(?, 2), dtype=float32

    return out
# Layer: Z[?,2] >> hidden1[?, 16] >> hidden2[?,16] >> out[?,2]

### Creates a fully connected neural network of 3 hidden layers
def createDiscriminatorNetWork(real_placeholder, hsize=[16, 16], reuse=False):
    with tf.variable_scope(DISCRIMINATOR_SCOPE,reuse=reuse):
        # dense(inputs, units, activation ...)
        hidden1 = tf.layers.dense(real_placeholder, hsize[0], activation=tf.nn.leaky_relu) # h1 Tensor("GAN/Discriminator/dense/LeakyRelu:0", shape=(?, 16), dtype=float32)
        hidden2 = tf.layers.dense(hidden1, hsize[1], activation=tf.nn.leaky_relu) #h2 Tensor("GAN/Discriminator/dense_1/LeakyRelu:0", shape=(?, 16), dtype=float32)
        hidden3 = tf.layers.dense(hidden2, 4) # h3 Tensor("GAN/Discriminator/dense_2/BiasAdd:0", shape=(?, 2), dtype=float32)
        out = tf.layers.dense(hidden3, 1) # out Tensor("GAN/Discriminator/dense_3/BiasAdd:0", shape=(?, 1), dtype=float32)

    return out
# Layer: X[?,2] >> hidden1[?, 16] >> hidden2[?,16] >> hidden3[?,2] >> out[?,1]

### Define Placeholders
real_samples_placeholder = tf.placeholder(tf.float32,[None,4]) # n by 2
noise_sample_placeholder = tf.placeholder(tf.float32,[None,4]) # n by 2

### Generator Neural Network
generator_network = createGeneratorNetowk(noise_sample_placeholder)
### Discriminator Neural Network for Real Sample Data
discriminator_real_network = createDiscriminatorNetWork(real_samples_placeholder)
### Discriminator Neural Network for Generator Sample Noise Data
discriminator_fake_network = createDiscriminatorNetWork(generator_network, reuse=True) # reuse: true >> generator network reuse

### Cost function
# tf.nn.sigmoid_cross_entropy_with_logits: Cross Entropy
cost_real_discriminator = tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_real_network,labels=tf.ones_like(discriminator_real_network))
cost_fake_discriminator = tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_fake_network,labels=tf.zeros_like(discriminator_fake_network))
cost_discriminator = tf.reduce_mean(cost_real_discriminator+cost_fake_discriminator)
cost_generator = tf.nn.sigmoid_cross_entropy_with_logits(logits=discriminator_fake_network,labels=tf.ones_like(discriminator_fake_network))
cost_generator = tf.reduce_mean(cost_generator)

### Variables collection
# variable_scope과 get_variable()함수의 조합은 name filed의 String 값을 알고 있어야 사용 가능
# collection과 tf.get_collection(key, scope)의 조합으로 변수로 활용 가능
vars_generator = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=GENERATOR_SCOPE)
vars_discriminator = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=DISCRIMINATOR_SCOPE)

### Optimization: RMSPropOptimizer
# tf.train.RMSPropOptimizer: mini-batch gradient descent
optimizer_generator = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost_generator, var_list = vars_generator)
optimizer_discriminator = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(cost_discriminator, var_list = vars_discriminator)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

steps_discriminator = 10
steps_generator = 6

### Write LossLog File
f = open(ts.getOutputPath()+'loss_logs.csv','w')
f.write('Iteration,Discriminator Loss,Generator Loss\n')

for i in range(10001):
    real_batch = createMatrix(cA3,cD3,cD2,cD1)
    noise_batch = createMatrix(fake_cA3,fake_cD3,fake_cD2,fake_cD1)
    # noise_batch = np.random.rand(4, cD1.size)

    for _ in range(steps_discriminator):
        _, loss_discriminator = sess.run([optimizer_discriminator, cost_discriminator], feed_dict={real_samples_placeholder: real_batch, noise_sample_placeholder: noise_batch})

    for _ in range(steps_generator):
        _, loss_generator = sess.run([optimizer_generator, cost_generator], feed_dict={noise_sample_placeholder: noise_batch})

    print ("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i, loss_discriminator, loss_generator))

    # Write log file
    if i%10 == 0:
        f.write("%d,\t\t%f,\t\t%f\n"%(i, loss_discriminator, loss_generator))

    stepBreaker = 1000
    # Trace Figure
    if i%stepBreaker == 0:
        generatorSummary = sess.run(generator_network, feed_dict={noise_sample_placeholder: noise_batch})
        # realPos = plt.scatter(real_pos[:,0], real_pos[:,1])
        # generatorPos = plt.scatter(generatorSummary[:,0],generatorSummary[:,1])
        print(np.array(generatorSummary[:,1]).dtype);
        print(fake_cA3.dtype)
        gene_cA3 = np.array(generatorSummary[:,0])[0:cA3.size]
        gene_cD3 = np.array(generatorSummary[:,1])[0:cD3.size]
        gene_cD2 = np.array(generatorSummary[:,2])[0:cD2.size]
        gene_cD1 = np.array(generatorSummary[:,3])
        
        trainList = [gene_cA3, gene_cD3, gene_cD2, gene_cD1]
        ts.traceFigure(targetList=trainList, targetRate=samplingRate, index=(i/stepBreaker), targetWavelet='db2', realData=realData)

f.close()



