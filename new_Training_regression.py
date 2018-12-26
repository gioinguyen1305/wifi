import tensorflow as tf
import numpy as np
import Path_new
import tools_new
import new_regression_model

training_data_path  = 'D:\\CCU\\ANguyen\\Environment\\Museum\\Processed_Data\\Code\\route_all\\route_all.csv'
training_label_path = 'D:\\CCU\\ANguyen\\Environment\\Museum\\Processed_Data\\Code\\route_all\\route_all_label_240.csv'
training_data = np.genfromtxt(training_data_path, delimiter=",")
training_label = np.genfromtxt(training_label_path, delimiter=",")
#region_matrix = np.genfromtxt('D:\\CCU\\ANguyen\\Environment\\Museum\\Processed_Data\\Code\\route_1\\matrix_1.csv', delimiter=",")

auto_model_path = 'D:\\CCU\\ANguyen\\Report\\Python\\Train\\Model_%d.npy' % (126000)


# Using mean and standard deviation to normalize the data
normalized_data, m, std = tools_new.normalize_data4(training_data)
normalize_parameter = np.array([m, std])
np.save('D:\\CCU\\ANguyen\\Report\\Python\\Train\\normalize_parameter.npy', normalize_parameter)

regress = new_regression_model.All(auto_model_path, None)
regress.build()

# ====== Constant ======
EPOCH = 5001
BATCH_SIZE = 240
TOTAL_SAMPLE_NUM = training_data.shape[0]
LOCATION_SAMPLE_NUM = 240
LOCATION_NUM = TOTAL_SAMPLE_NUM/LOCATION_SAMPLE_NUM
BATCH = int(TOTAL_SAMPLE_NUM/BATCH_SIZE)
Set = np.arange(TOTAL_SAMPLE_NUM, dtype='i')
Min_loss = 100
Min_loss_step = 0
# ======================

with tf.Session() as sess:

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('D:/CCU/ANguyen/Report/Python/Train/regression/')
    writer.add_graph(sess.graph)

    sess.run(regress.initial())

    for epoch in range(0, EPOCH):
        np.random.shuffle(Set)
        for batch in range(0, BATCH):
            batch_set = Set[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
            batch_data, batch_label = tools_new.next_batch(normalized_data, training_label, batch_set)
            _, cost = sess.run([regress.optimizer, regress.loss], feed_dict={regress.x: batch_data,
                                                                             regress.gt: batch_label})

            step = epoch * BATCH + batch

            if np.isnan(cost):
                print ('NaN')
                quit()

            if step % 1 == 0:
                print ('(Step:%d) epoch %d batch %d: loss %.3f' % (step, epoch, batch, cost))
                rs = sess.run(merged, feed_dict={regress.x:batch_data,
                                                 regress.gt: batch_label})

                writer.add_summary(rs, step)

            if step % 2000 == 0 and step > 0:
                model_path = 'D:\\CCU\\ANguyen\\Report\\Python\Train\\regression\\Model_%d.npy' % (step)
                regress.savemodel(model_path)
                print ("Save model !!")
                if Min_loss > cost:
                    Min_loss = cost
                    Min_loss_step = step
    print(Min_loss_step)