import tensorflow as tf
import numpy as np # 파이썬에서 배열을 사용하기 위한 표준 패키지
import matplotlib.pyplot as plt # Matplotlib는 파이썬에서 자료를 차트(chart)나 플롯(plot)으로 시각화(visulaization)하는 패키지
import os
import uuid

# Output Directory
ROOT_RESULT_PATH = os.path.abspath(__file__ + "../../../../")+'/output/plots/'
GENERATOR_SCOPE = "GAN/Generator"
DISCRIMINATOR_SCOPE = "GAN/Discriminator"

### Create Real Sample Data
def createRealSample(n=10000, scale=100):
    data = []

    # np.random.random_sample: 0.0 ~ 1.0 random variable
    # 5 * np.random.random_sample((3, 2)) - 5 >> [-5 ~ 0]
    # 3 by 2 >> array([[-3.99149989, -0.52338984], [-2.99091858, -0.79479508], [-1.23204345, -1.75224494]])
    x = scale*(np.random.random_sample((n,))-0.5) # n by 1, (-50 ~ 50 )

    for i in range(n):
        yi = 10 + x[i] * x[i]
        data.append([x[i], yi])
    # n by 2, (10 ~ 2510) >> [[8.024382387854756, 74.39071270651358], [-1.9397242335024378, 13.76253010203662], ..., [28.81166337629969, 840.1119465092089]]

    return np.array(data)

### Creates a fully connected neural network of 2 hidden layers
def createGeneratorNetowk(noise_placeholder, hsize=[16, 16], reuse=False): # Z: [none, 2]
    with tf.variable_scope(GENERATOR_SCOPE,reuse=reuse):
        # dense(inputs, units, activation ...)
        hidden1 = tf.layers.dense(noise_placeholder, hsize[0], activation=tf.nn.leaky_relu) # hidden1 Tensor name: GAN/Generator/dense/LeakyRelu:0, shape=(?, 16), dtype=float32
        hidden2 = tf.layers.dense(hidden1, hsize[1], activation=tf.nn.leaky_relu) # hidden2 name: GAN/Generator/dense_1/LeakyRelu:0, shape=(?, 16), dtype=float32
        out = tf.layers.dense(hidden2, 2) # out name: GAN/Generator/dense_2/BiasAdd:0, shape=(?, 2), dtype=float32

    return out
# Layer: Z[?,2] >> hidden1[?, 16] >> hidden2[?,16] >> out[?,2]

### Creates a fully connected neural network of 3 hidden layers
def createDiscriminatorNetWork(real_placeholder, hsize=[16, 16], reuse=False):
    with tf.variable_scope(DISCRIMINATOR_SCOPE,reuse=reuse):
        # dense(inputs, units, activation ...)
        hidden1 = tf.layers.dense(real_placeholder, hsize[0], activation=tf.nn.leaky_relu) # h1 Tensor("GAN/Discriminator/dense/LeakyRelu:0", shape=(?, 16), dtype=float32)
        hidden2 = tf.layers.dense(hidden1, hsize[1], activation=tf.nn.leaky_relu) #h2 Tensor("GAN/Discriminator/dense_1/LeakyRelu:0", shape=(?, 16), dtype=float32)
        hidden3 = tf.layers.dense(hidden2, 2) # h3 Tensor("GAN/Discriminator/dense_2/BiasAdd:0", shape=(?, 2), dtype=float32)
        out = tf.layers.dense(hidden3, 1) # out Tensor("GAN/Discriminator/dense_3/BiasAdd:0", shape=(?, 1), dtype=float32)

    return out
# Layer: X[?,2] >> hidden1[?, 16] >> hidden2[?,16] >> hidden3[?,2] >> out[?,1]

### Define Placeholders
real_samples_placeholder = tf.placeholder(tf.float32,[None,2]) # n by 2
noise_sample_placeholder = tf.placeholder(tf.float32,[None,2]) # n by 2

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

### TODO: Tensor board
# tf.summary.scalar("Discriminator cost", cost_discriminator);
# tf.summary.scalar("Generator cost", cost_generator);
# merged_summary_op = tf.summary.merge_all()
# # writer = tf.summary.FileWriter("/tmp/test_logs", sess.graph)
# uniq_id = "/tmp/tensorboard-layers-api/" + uuid.uuid1().__str__()[:6]
# summary_writer = tf.summary.FileWriter(uniq_id, graph=tf.get_default_graph())

batch_size = 256
steps_discriminator = 10
steps_generator = 6

real_pos = createRealSample(n=batch_size)

### Write LossLog File
if not os.path.exists(ROOT_RESULT_PATH):
    os.makedirs(ROOT_RESULT_PATH)
    os.makedirs(ROOT_RESULT_PATH+'iterations/')
f = open(ROOT_RESULT_PATH+'loss_logs.csv','w')
f.write('Iteration,Discriminator Loss,Generator Loss\n')

for i in range(10001):
    real_batch = createRealSample(n=batch_size) # 256 by 2 >> [[8.024382387854756, 74.39071270651358], [-1.9397242335024378, 13.76253010203662], ..., [28.81166337629969, 840.1119465092089]]
    noise_batch = np.random.uniform(-1.0, 1.0, size=[batch_size, 2])  # 256 by 2, (-1 ~ 1) >> [[0.96149095  0.25940196] [-0.90235707 -0.58915083] ... [-0.44557393 -0.25887667]]

    for _ in range(steps_discriminator): # _ : ignore index
        _, loss_discriminator = sess.run([optimizer_discriminator, cost_discriminator], feed_dict={real_samples_placeholder: real_batch, noise_sample_placeholder: noise_batch})

    for _ in range(steps_generator):
        _, loss_generator = sess.run([optimizer_generator, cost_generator], feed_dict={noise_sample_placeholder: noise_batch})

    print ("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i, loss_discriminator, loss_generator))

    # Write log file
    if i%10 == 0:
        f.write("%d,\t\t%f,\t\t%f\n"%(i, loss_discriminator, loss_generator))

    # Draw Graph
    if i%1000 == 0:
        plt.figure()
        generatorSummary = sess.run(generator_network, feed_dict={noise_sample_placeholder: noise_batch})
        realPos = plt.scatter(real_pos[:,0], real_pos[:,1])
        generatorPos = plt.scatter(generatorSummary[:,0],generatorSummary[:,1])

        plt.legend((realPos,generatorPos), ("Real Data","Generated Data"))
        plt.title('Samples at Iteration %d'%i)
        plt.tight_layout()
        plt.savefig(ROOT_RESULT_PATH+'iterations/iteration_%d.png'%i)
        plt.close()

f.close()