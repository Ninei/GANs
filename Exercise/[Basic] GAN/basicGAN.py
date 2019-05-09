import tensorflow as tf
import numpy as np # 파이썬에서 배열을 사용하기 위한 표준 패키지
import seaborn # Seaborn은 Matplotlib을 기반으로 다양한 색상 테마와 통계용 차트 등의 기능을 추가한 시각화 패키지
import matplotlib.pyplot as plt # Matplotlib는 파이썬에서 자료를 차트(chart)나 플롯(plot)으로 시각화(visulaization)하는 패키지
import os

# 최종 결과 파일 저장을 위한 디렉토리
ROOT_RESULT_PATH = os.path.abspath(__file__ + "../../../../")+'/output/plots/'

# 샘플 데이터 생성
def createSampleData(n=10000, scale=100):
    data = []

    # np.random.random_sample: 0.0 ~ 1.0 random variable
    # 5 * np.random.random_sample((3, 2)) - 5 >> [-5 ~ 0]
    # 3 by 2 >> array([[-3.99149989, -0.52338984], [-2.99091858, -0.79479508], [-1.23204345, -1.75224494]])
    x = scale*(np.random.random_sample((n,))-0.5) # n by 1 >>  array([49.149989 -1.5224494 ... 2.5324167])

    for i in range(n):
        yi = 10 + x[i] * x[i]
        data.append([x[i], yi])
    # n by 2 >> [[8.024382387854756, 74.39071270651358], [-1.9397242335024378, 13.76253010203662], ..., [28.81166337629969, 840.1119465092089]]

    return np.array(data)

# Creates a fully connected neural network of 2 hidden layers
def generator(Z, hsize=[16, 16], reuse=False): # Z: [none, 2]
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        # dense(inputs, units, activation ...), Outputs = activation(inputs * kernel + bias)
        hidden1 = tf.layers.dense(Z, hsize[0], activation=tf.nn.leaky_relu) # hidden1 Tensor name: GAN/Generator/dense/LeakyRelu:0, shape=(?, 16), dtype=float32
        hidden2 = tf.layers.dense(hidden1, hsize[1], activation=tf.nn.leaky_relu) # hidden2 name: GAN/Generator/dense_1/LeakyRelu:0, shape=(?, 16), dtype=float32
        out = tf.layers.dense(hidden2, 2) # out name: GAN/Generator/dense_2/BiasAdd:0, shape=(?, 2), dtype=float32

    return out
# Layer: Z[?,2] >> hidden1[?, 16] >> hidden2[?,16] >> out[?,2]

# Creates a fully connected neural network of 3 hidden layers
def discriminator(X, hsize=[16, 16], reuse=False):
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        hidden1 = tf.layers.dense(X, hsize[0], activation=tf.nn.leaky_relu) # h1 Tensor("GAN/Discriminator/dense/LeakyRelu:0", shape=(?, 16), dtype=float32)
        hidden2 = tf.layers.dense(hidden1, hsize[1], activation=tf.nn.leaky_relu) #h2 Tensor("GAN/Discriminator/dense_1/LeakyRelu:0", shape=(?, 16), dtype=float32)
        hidden3 = tf.layers.dense(hidden2, 2) # h3 Tensor("GAN/Discriminator/dense_2/BiasAdd:0", shape=(?, 2), dtype=float32)
        out = tf.layers.dense(hidden3, 1) # out Tensor("GAN/Discriminator/dense_3/BiasAdd:0", shape=(?, 1), dtype=float32)

    return out, hidden3
# Layer: X[?,2] >> hidden1[?, 16] >> hidden2[?,16] >> hidden3[?,2] >> out[?,1]

# Set aesthetic parameters in one step.
seaborn.set()

X = tf.placeholder(tf.float32,[None,2]) # n by 2
Z = tf.placeholder(tf.float32,[None,2]) # n by 2


# 왜 이렇게 하는지.....
G_sample = generator(Z)
r_logits, r_rep = discriminator(X) # r_logits: out, r_rep: hidden3
f_logits, g_rep = discriminator(G_sample, reuse=True) # f_logits: out, g_rep: hidden3
# reuse: true >> https://github.com/tensorflowkorea/tensorflow-kr/blob/master/g3doc/how_tos/variable_scope/index.md

# 손실함수 정의
# tf.reduce_mean: ??
# tf.nn.sigmoid_cross_entropy_with_logits: ??
# tf.ones_like: tensor와 동일한 타입과 형태의 텐서를 만들고, 모든 원소의 값을 1로 초기화
# tf.zeros_like: tensor와 동일한 타입과 형태의 텐서를 만들고, 모든 원소의 값을 0으로 초기화
loss_discriminator = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
loss_generator = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))

# Variables collection 설정
# 해당 variable을 코드의 다른 위치에서 불러오기 위해, variable_scope과 get_variable()함수의 조합은 name filed값을 기억 사용 가능
# 특정 목적을 위한 variable의 집합을 불러올 때는 collection과 tf.get_collection()의 조합으로 가능
# tf.get_collection(key)가 실행되면, key의 collection에 속하는 variable들의 리스트가 리턴
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

# 최적화: RMSPropOptimizer
# tf.train.RMSPropOptimizer: mini-batch gradient descent(Divide the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight.)
# minimize: Add operations to minimize loss by updating var_list.
train_generator = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss_generator, var_list = gen_vars) # G Train step
train_discriminator = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss_discriminator, var_list = disc_vars) # D Train step

# sess = tf.Session(config=config)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

batch_size = 256
nd_steps = 10
ng_steps = 10

x_plot = createSampleData(n=batch_size)

# Write LossLog File
f = open(ROOT_RESULT_PATH+'loss_logs.csv','w')
f.write('Iteration,Discriminator Loss,Generator Loss\n')

for i in range(10001):
    X_batch = createSampleData(n=batch_size) # 256 by 2, n by 2 >> [[8.024382387854756, 74.39071270651358], [-1.9397242335024378, 13.76253010203662], ..., [28.81166337629969, 840.1119465092089]]
    Z_batch = np.random.uniform(-1.0, 1.0, size=[batch_size, 2])  # 256 by 2, (-1 ~ 0) >> [[0.96149095  0.25940196] [-0.90235707 -0.58915083] ... [-0.44557393 -0.25887667]]

    for _ in range(nd_steps):
        _, dloss = sess.run([train_discriminator, loss_discriminator], feed_dict={X: X_batch, Z: Z_batch})
    rrep_dstep, grep_dstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

    for _ in range(ng_steps):
        _, gloss = sess.run([train_generator, loss_generator], feed_dict={Z: Z_batch})

    rrep_gstep, grep_gstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

    print ("Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i,dloss,gloss))
    if i%10 == 0:
        f.write("%d,%f,%f\n"%(i,dloss,gloss))

    if i%1000 == 0:
        plt.figure()
        g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})
        xax = plt.scatter(x_plot[:,0], x_plot[:,1])
        gax = plt.scatter(g_plot[:,0],g_plot[:,1])

        plt.legend((xax,gax), ("Real Data","Generated Data"))
        plt.title('Samples at Iteration %d'%i)
        plt.tight_layout()
        plt.savefig(ROOT_RESULT_PATH+'iterations/iteration_%d.png'%i)
        plt.close()

        plt.figure()
        rrd = plt.scatter(rrep_dstep[:,0], rrep_dstep[:,1], alpha=0.5)
        rrg = plt.scatter(rrep_gstep[:,0], rrep_gstep[:,1], alpha=0.5)
        grd = plt.scatter(grep_dstep[:,0], grep_dstep[:,1], alpha=0.5)
        grg = plt.scatter(grep_gstep[:,0], grep_gstep[:,1], alpha=0.5)

        plt.legend((rrd, rrg, grd, grg), ("Real Data Before G step","Real Data After G step",
                               "Generated Data Before G step","Generated Data After G step"))
        plt.title('Transformed Features at Iteration %d'%i)
        plt.tight_layout()
        plt.savefig(ROOT_RESULT_PATH+'features/feature_transform_%d.png'%i)
        plt.close()

        plt.figure()

        rrdc = plt.scatter(np.mean(rrep_dstep[:,0]), np.mean(rrep_dstep[:,1]),s=100, alpha=0.5)
        rrgc = plt.scatter(np.mean(rrep_gstep[:,0]), np.mean(rrep_gstep[:,1]),s=100, alpha=0.5)
        grdc = plt.scatter(np.mean(grep_dstep[:,0]), np.mean(grep_dstep[:,1]),s=100, alpha=0.5)
        grgc = plt.scatter(np.mean(grep_gstep[:,0]), np.mean(grep_gstep[:,1]),s=100, alpha=0.5)

        plt.legend((rrdc, rrgc, grdc, grgc), ("Real Data Before G step","Real Data After G step",
                               "Generated Data Before G step","Generated Data After G step"))

        plt.title('Centroid of Transformed Features at Iteration %d'%i)
        plt.tight_layout()
        plt.savefig(ROOT_RESULT_PATH+'features/feature_transform_centroid_%d.png'%i)
        plt.close()

f.close()