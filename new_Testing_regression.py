
import tensorflow as tf
import numpy as np
import Path_new
import tools_new
import new_regression_model
import evaluate_tools
from matplotlib import pyplot as plt

training_data_path  = 'D:\\CCU\\ANguyen\\Environment\\Museum\\Processed_Data\\Code\\route_all\\route_all.csv'
training_label_path = 'D:\\CCU\\ANguyen\\Environment\\Museum\\Processed_Data\\Code\\route_all\\route_all_label_240.csv'
training_data = np.genfromtxt(training_data_path, delimiter=",")
training_label = np.genfromtxt(training_label_path, delimiter=",")

testing_data_path, testing_label_path, route = Path_new.path(2,8,1)
testing_data =              np.genfromtxt(testing_data_path, delimiter=",")
testing_label =             np.genfromtxt(testing_label_path, delimiter=",")

normalize_parameter =       np.load('D:\\CCU\\ANguyen\\Report\\Python\\Train\\regression\\normalize_parameter.npy')

normalized_training_data = tools_new.normalize_data5(training_data, normalize_parameter[0], normalize_parameter[1])
normalized_testing_data =  tools_new.normalize_data5(testing_data, normalize_parameter[0], normalize_parameter[1])

for i in range(74, 75):
    iteration = int(i*2000)
    model_path = 'D:\\CCU\\ANguyen\\Report\\Python\Train\\regression\\Model_%d.npy' % (iteration)
    regress = new_regression_model.All(None, model_path)
    regress.build()
    with tf.Session() as sess:
        sess.run(regress.initial())
        training_predict = sess.run([regress.x_regression2], feed_dict={regress.x: normalized_training_data})
        testing_predict = sess.run([regress.x_regression2], feed_dict={regress.x: normalized_testing_data})
        testing_predict_temp = testing_predict[0]
        training_predict_temp = training_predict[0]
        
        f = plt.figure()
        ax = f.add_subplot(111)
        
        ax.scatter(testing_predict_temp[:,0],testing_predict_temp[:,1],10,color='red',label = 'Inferred')
        ax.plot(testing_label[:,0], testing_label[:,1], 'b-o',label = 'Label')
        ax.legend(loc='upper left', numpoints = 1)
        ax.set_aspect(0.6)
        plt.show()
        np.save('D:/CCU/ANguyen/Report/Python/DNN/Route_8/Inferred_label.npy',testing_predict_temp)
        np.save('D:/CCU/ANguyen/Report/Python/DNN/Route_8/Ground_Truth_label.npy',testing_label)
        
        training_d_err = evaluate_tools.distance_err(training_predict_temp, training_label)
        testing_d_err = evaluate_tools.distance_err(testing_predict_temp, testing_label)

        a = evaluate_tools.all(training_d_err)
        b = evaluate_tools.all(testing_d_err)
        np.save('D:/CCU/ANguyen/Report/Python/DNN/Route_8/dis_err_matlab.npy',b)
        print ('Training/ index %d mean: %.6f max: %.6f var: %.6f' % (iteration, a[0], a[2], a[1]))
        print ('Testing/ index %d mean: %.6f max: %.6f var: %.6f' % (iteration, b[0], b[2], b[1]))


        #train_x, train_y = evaluate_tools.cdf(training_d_err)
        #test_x, test_y = evaluate_tools.cdf(testing_d_err)

        #plt.figure()
        #plt.xlabel('Distance Error')
        #plt.ylabel('Probability')
        #plt.title('CDF in a offical environment')

        #plt.xlim(0, 15, 1)
        #plt.ylim(0, 1.1, 0.1)

        #plt.plot(train_x, train_y, 'r-*', linewidth=2.0)
        #plt.plot(test_fast_x, test_fast_y, 'g-*', linewidth=2.0)
        #plt.plot(test_slow_x, test_slow_y, 'b-*', linewidth=2.0)

        #plt.legend(labels=['Training', 'Testing_fast', 'Testing_slow'])
        #plt.show()

        #np.save(training_feature_path, training_fc3)
        #np.save(testing_feature_fast_path, testing_fast_fc3)
        #np.save(testing_feature_slow_path, testing_slow_fc3)
