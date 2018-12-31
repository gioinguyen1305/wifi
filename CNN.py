from __future__ import division, print_function, absolute_import
import ast
import tensorflow as tf
from tensorflow.contrib import rnn
import csv
import pandas as pd
import numpy as np
import tools_new
with open('day.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    temps = []
    atemps = []
    hums = []
    windspeeds = []
    for row in readCSV:
        temp = ast.literal_eval(row[5])
        atemp = ast.literal_eval(row[6])
        hum = ast.literal_eval(row[7])
        windspeed = ast.literal_eval(row[8])
        temps.append([temp])
        hums.append([hum])
        atemps.append([hum])
        windspeeds.append([windspeed])

data = np.concatenate((np.asarray(temps),np.asarray(atemps),np.asarray(windspeeds)), axis=1)
data = np.asarray(data).reshape(731,3,1)
label = np.asarray(hums)
data_train = data[:700]
label_train = label[:700]
data_test = data[700:]
label_test = label[700:]

# Training Parameters
learning_rate = 0.001
num_steps = 200
batch_size = 128
display_step = 10

dropout = 0.75 # Dropout, probability to keep units

# tf Graph input

X = tf.placeholder("float", [None, 3, 1])
Y = tf.placeholder("float", [None, 1])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([3,1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, 1]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([1]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.reduce_sum(tf.square(logits - Y), 1)) 
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss_op)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1, epoch+1):
        np.random.shuffle(Set)
        for b in range(0,Batch):
            batch_set = Set[b*batch_size:(b+1)*batch_size]
            batch_data, batch_label = tools_new.next_batch(data_train,label_train,batch_set)
            sess.run(train_op, feed_dict={X: batch_data, Y: batch_label})

            if step % 10 == 0 or step == 1:
                loss, pre= sess.run([loss_op,logits], feed_dict={X: batch_data,     Y: batch_label,keep_prob: 1.0})
                acc = 1-(np.square(np.abs(pre - batch_label)) ).mean()
                print ('Step:%d   Batch:%d   Minibatch Loss: %.8f   Accuracy: %.8f ' % (step,b+1,loss,acc))
    print ("Test!")
    loss, pre= sess.run([loss_op,logits], feed_dict={X: data_test,     Y: label_test})
    acc = 1-(np.square(np.abs(pre - label_test)) ).mean()
    print ('Loss: %.8f   Accuracy: %.8f ' % (loss,acc))