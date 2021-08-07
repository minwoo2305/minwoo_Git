import input_data
import tensorflow as tf
from Early_Stopping import EarlyStopping
from Genetic_Algorithm import genetic_algorithm
import numpy as np
import time

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784, 200], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_normal([200, 50], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_normal([50, 10], stddev=0.01))
model = tf.matmul(L2, W3)

W_list = [W1, W2, W3]

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

# 시작시간 저장
start = time.time()
early_stopping = EarlyStopping()

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# GA test
with sess.as_default():
    for weight in enumerate(W_list):
        W_list[weight[0]] = weight[1].eval()

    ga = genetic_algorithm(W_list, num_of_population=10, extend_range=0.01)
    chromosomes = ga.make_population()

    # Generation Test
    best_chromosome = chromosomes[0]
    for generation in range(10):
        print("============================= generation " + str(generation + 1) + " =============================")
        for chromosome in enumerate(chromosomes):
            sess.run(init)
            for case in enumerate(chromosome[1][:-1]):
                W_list[case[0]] = case[1]
            acc = sess.run(accuracy, feed_dict={X: mnist.train.images, Y: mnist.train.labels})
            ga.set_fitness(chromosome[0], acc)

        chromosomes, first_chromosome = ga.generation(10 - generation)

        if best_chromosome[-1] < first_chromosome[-1]:
            best_chromosome = first_chromosome

    for case in enumerate(best_chromosome[:-1]):
        W_list[case[0]] = case[1]

print(sess.run(accuracy, feed_dict={X: mnist.train.images, Y: mnist.train.labels}))

for epoch in range(50):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    if early_stopping.validate_mean(total_cost/total_batch, patience=5):
        break

    print('Epoch :' + str(epoch + 1) + '  Avg cost : ' + str(total_cost / total_batch))

print('최적화 완료!')

print("time : ", time.time() - start)

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
