
from __future__ import print_function
import ast
import tensorflow as tf
from tensorflow.contrib import rnn
import csv
import pandas as pd
import numpy as np
import tools_new
from sklearn.metrics import accuracy_score
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
epoch = 3500
batch_size = 700
total = label_train.shape[0]
Set = np.arange(total, dtype='i')
Batch = int(total/batch_size)

X = tf.placeholder("float", [None, 3, 1])
Y = tf.placeholder("float", [None, 1])

weights = {'out': tf.Variable(tf.random_normal([100, 1]))}
biases = {'out': tf.Variable(tf.random_normal([1]))}


def RNN(x, weights, biases):
    x = tf.unstack(x, 3, 1)
    rnn_cell = tf.contrib.rnn.BasicRNNCell(100)
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)

loss_op = tf.reduce_mean(tf.reduce_sum(tf.square(logits - Y), 1)) 
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss_op)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1, epoch+1):
        np.random.shuffle(Set)
        for b in range(0,Batch):
            batch_set = Set[b*batch_size:(b+1)*batch_size]
            batch_data, batch_label = tools_new.next_batch(data_train,label_train,batch_set)
            sess.run(train_op, feed_dict={X: batch_data, Y: batch_label})

            if step % 10 == 0 or step == 1:
                loss, pre= sess.run([loss_op,logits], feed_dict={X: batch_data,     Y: batch_label})
                acc = 1-(np.square(np.abs(pre - batch_label)) ).mean()
                print ('Step:%d   Batch:%d   Minibatch Loss: %.8f   Accuracy: %.8f ' % (step,b+1,loss,acc))
    print ("Test!")
    loss, pre= sess.run([loss_op,logits], feed_dict={X: data_test,     Y: label_test})
    acc = 1-(np.square(np.abs(pre - label_test)) ).mean()
    print ('Loss: %.8f   Accuracy: %.8f ' % (loss,acc))
