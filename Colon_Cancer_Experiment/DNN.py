import tensorflow as tf
import numpy as np

data = np.genfromtxt('Train.csv', delimiter=',', dtype=np.float32, encoding='utf-8', skip_header=1)
data_set = data[:, :-2]
label_set = data[:, -2:]

test_data_set = np.genfromtxt('Test.csv', delimiter=',', dtype=np.float32, encoding='utf-8', skip_header=1)

'''
train_data = data_set[:16082]
train_label = label_set[:16082]
test_data = data_set[16082:]
test_label = label_set[16082:]
'''

train_data = data_set[:]
train_label = label_set[:]
test_data = test_data_set[:, :-2]
test_label = test_data_set[:, -2:]

X = tf.placeholder(tf.float32, [None, 7])
Y = tf.placeholder(tf.float32, [None, 2])

W1 = tf.Variable(tf.random_normal([7, 7], stddev=0.01))
b1 = tf.Variable(tf.random_normal([7]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([7, 7], stddev=0.01))
b2 = tf.Variable(tf.random_normal([7]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([7, 2], stddev=0.01))
b3 = tf.Variable(tf.random_normal([2]))
model = tf.matmul(L2, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
specificity_list = []

for count in range(30):
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print('=============================== Step ' + str(count + 1) + ' ===============================')
    for epoch in range(10):
        total_cost = 0

        for step in range(100):
            _, cost_val = sess.run([optimizer, cost], feed_dict={X: train_data, Y: train_label})
            total_cost += cost_val

        print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.5f}'.format(total_cost / 10))

    print('최적화 완료!')

    predicted = tf.argmax(model, 1)
    target = tf.argmax(Y, 1)

    tp = tf.count_nonzero(predicted * target)
    tn = tf.count_nonzero((predicted - 1) * (target - 1))
    fp = tf.count_nonzero(predicted * (target - 1))
    fn = tf.count_nonzero((predicted - 1) * target)

    accuracy = ((tp + tn) / (tp + tn + fp + fn))
    precision = (tp / (tp + fp))
    # recall = sensitivity
    recall = (tp / (tp + fn))
    f1 = ((2 * precision * recall) / (precision + recall))
    specificity = (tn / (tn + fp))

    tp_val = sess.run(tp, feed_dict={X: test_data, Y: test_label})
    tn_val = sess.run(tn, feed_dict={X: test_data, Y: test_label})
    fp_val = sess.run(fp, feed_dict={X: test_data, Y: test_label})
    fn_val = sess.run(fn, feed_dict={X: test_data, Y: test_label})

    train_accuracy_val = sess.run(accuracy, feed_dict={X: train_data, Y: train_label})
    test_accuracy_val = sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
    precision_val = sess.run(precision, feed_dict={X: test_data, Y: test_label})
    # recall = sensitivity
    recall_val = sess.run(recall, feed_dict={X: test_data, Y: test_label})
    f1_val = sess.run(f1, feed_dict={X: test_data, Y: test_label})
    specificity_val = sess.run(specificity, feed_dict={X: test_data, Y: test_label})

    print('TP : ' + str(tp_val))
    print('TN : ' + str(tn_val))
    print('FP : ' + str(fp_val))
    print('FN : ' + str(fn_val))
    print('----------------------------------------------------------------------')
    print('Train Accuracy : ' + str(train_accuracy_val))
    print('Test Accuracy : ' + str(test_accuracy_val))
    print('Precision : ' + str(precision_val))
    print('Recall : ' + str(recall_val))
    print('Specificity : ' + str(specificity_val))
    print('F1 Score : ' + str(f1_val))
    print('----------------------------------------------------------------------')
    print('\n\n')

    accuracy_list.append(test_accuracy_val)
    precision_list.append(precision_val)
    recall_list.append(recall_val)
    f1_list.append(f1_val)
    specificity_list.append(specificity_val)

print('Avg. Accuracy : ' + str(np.mean(accuracy_list)) + ', Std. Accuracy : ' + str(np.std(accuracy_list)))
print('Avg. Precision : ' + str(np.mean(precision_list)) + ', Std. Precision : ' + str(np.std(precision_list)))
print('Avg. Recall : ' + str(np.mean(recall_list)) + ', Std. Recall : ' + str(np.std(recall_list)))
print('Avg. F1 Score : ' + str(np.mean(f1_list)) + ', Std. F1 Score : ' + str(np.std(f1_list)))
print('Avg. Specificity : ' + str(np.mean(specificity_list)) + ', Std. Specificity : ' + str(np.std(specificity_list)))
