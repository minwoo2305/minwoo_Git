import tensorflow as tf

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0, 1], [1, 0], [1, 0], [0, 1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 2])

W = tf.Variable(tf.random_normal([2, 2]), name='weight')
b = tf.Variable(tf.random_normal([2]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})

        if step % 2000 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("\n---Hypothesis--- \n", h, "\n---Correct(Y)---\n", c, "\n---Accuracy---\n", a)