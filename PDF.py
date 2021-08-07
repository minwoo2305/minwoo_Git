import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

data_size = 100
noise_size = 10
mean = 7
std = 1
learning_rate = 0.00001
total_epoch = 15001

x = np.linspace(0, 10, data_size)
real = scipy.stats.norm(mean, std).pdf(x)
plt.plot(x, real, 'b')
plt.show()

real = np.array(real).reshape(1, data_size)

real_data = tf.placeholder(tf.float32, shape=(None, data_size))
z = tf.placeholder(tf.float32, shape=(None, noise_size))

G_W1 = tf.Variable(tf.random_normal([noise_size, 30], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([30]))
G_W2 = tf.Variable(tf.random_normal([30, data_size], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([data_size]))

D_W1 = tf.Variable(tf.random_normal([data_size, 10], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([10]))
D_W2 = tf.Variable(tf.random_normal([10, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))


def generator(noise):
    hidden = tf.nn.sigmoid(tf.matmul(noise, G_W1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)
    return output


def discriminator(input):
    hidden = tf.nn.sigmoid(tf.matmul(input, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)
    return output


def get_noise():
    return np.random.normal(size=(1, noise_size))


fake_data = generator(z)
D_gene = discriminator(fake_data)
D_real = discriminator(real_data)

loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))
loss_G = tf.reduce_mean(tf.log(D_gene))

D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    noise = get_noise()
    _, loss_val_D = sess.run([train_D, loss_D], feed_dict={real_data: real, z: noise})
    _, loss_val_G = sess.run([train_G, loss_G], feed_dict={z: noise})
    g = sess.run([fake_data], feed_dict={z: noise})

    if epoch % (total_epoch // 10) == 0:
        print('Epoch:', '%4d' % epoch, 'D loss: {:.4}'.format(loss_val_D), 'G loss: {:.4}'.format(loss_val_G))
        print(sess.run([D_real], feed_dict={real_data: real}))
        print(sess.run([D_gene], feed_dict={z: noise}))
        fake = np.array(g[0]).reshape(data_size, -1)
        real_d = np.array(real).reshape(data_size, -1)
        plt.plot(x, real_d, 'b')
        plt.plot(x, fake, 'r')
        plt.show()

print('최적화 완료!')
