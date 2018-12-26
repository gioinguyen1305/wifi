import tensorflow as tf
import numpy as np

# Autoencoder (initial weights) + regression


class All:
    def __init__(self, pretrain_model_path, model_path):

        input_nodes = 5
        label_nodes = 2
        ## input layer ##
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [None, input_nodes], name='x')
            
            self.gt = tf.placeholder(tf.float32, [None, label_nodes], name='gt')
            tf.summary.histogram('input/x', self.x)

        if model_path == None:
            pretrain_model = np.load(pretrain_model_path).item()

            input_nodes = 5
            layer1_nodes = 15
            layer2_nodes = 10
            layer3_nodes = 5
            label_nodes = 2

            self.weights = {
                'fc1_w': tf.Variable(tf.constant(pretrain_model["weights"]["encoder1_w"]), trainable=False, name='fc1_w'),
                'fc2_w': tf.Variable(tf.constant(pretrain_model["weights"]["encoder2_w"]), trainable=False, name='fc2_w'),
                'fc3_w': tf.Variable(tf.constant(pretrain_model["weights"]["encoder3_w"]), name='fc3_w'),
                'regression1_w': self.weight([layer3_nodes, layer3_nodes], 'regression1'),
                'regression2_w': self.weight([layer3_nodes, label_nodes], 'regression2')
            }

            self.biases = {
                'fc1_b': tf.Variable(tf.constant(pretrain_model["biases"]["encoder1_b"]), trainable=False, name='fc1_b'),
                'fc2_b': tf.Variable(tf.constant(pretrain_model["biases"]["encoder2_b"]), trainable=False, name='fc2_b'),
                'fc3_b': tf.Variable(tf.constant(pretrain_model["biases"]["encoder3_b"]), name='fc3_b'),
                'regression1_b': self.bias([layer3_nodes], 'regression1'),
                'regression2_b': self.bias([label_nodes], 'regression2')
            }

        if model_path != None:

            restore_model = np.load(model_path,encoding = 'latin1').item()

            self.weights = {
                'fc1_w': tf.Variable(tf.constant(restore_model["weights"]["fc1_w"]), name='fc1_w'),
                'fc2_w': tf.Variable(tf.constant(restore_model["weights"]["fc2_w"]), name='fc2_w'),
                'fc3_w': tf.Variable(tf.constant(restore_model["weights"]["fc3_w"]), name='fc3_w'),
                'regression1_w': tf.Variable(tf.constant(restore_model["weights"]["regression1_w"]), name='regression1_w'),
                'regression2_w': tf.Variable(tf.constant(restore_model["weights"]["regression2_w"]), name='regression2_w')
            }
            self.biases = {

                'fc1_b': tf.Variable(tf.constant(restore_model["biases"]["fc1_b"]), name='fc1_b'),
                'fc2_b': tf.Variable(tf.constant(restore_model["biases"]["fc2_b"]), name='fc2_b'),
                'fc3_b': tf.Variable(tf.constant(restore_model["biases"]["fc3_b"]), name='fc3_b'),
                'regression1_b': tf.Variable(tf.constant(restore_model["biases"]["regression1_b"]), name='regression1_b'),
                'regression2_b': tf.Variable(tf.constant(restore_model["biases"]["regression2_b"]), name='regression2_b')
            }

    def weight(self, shape, layername):
        W = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.3, dtype=tf.float32), name=layername + '_w')
        return W

    def bias(self, shape, layername):
        B = tf.Variable(tf.constant(0.001, shape=shape, dtype=tf.float32), name=layername + '_b')
        return B

    def initial(self):
        # init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        return init

    def fc_layer(self, inputs, w, b, activation_function=tf.nn.tanh):
        Wx_plus_b = tf.nn.bias_add(tf.matmul(inputs, w), b)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    def build(self):

        with tf.name_scope('fc1'):
            x_encoder1 = self.fc_layer(self.x, self.weights['fc1_w'], self.biases['fc1_b'], activation_function=tf.nn.tanh)
            tf.summary.histogram('fc1/output',x_encoder1)
            tf.summary.histogram('fc1/weights',self.weights['fc1_w'])
            tf.summary.histogram('fc1/biases', self.biases['fc1_b'])
        with tf.name_scope('fc2'):
            x_encoder2 = self.fc_layer(x_encoder1, self.weights['fc2_w'], self.biases['fc2_b'], activation_function=tf.nn.tanh)
            tf.summary.histogram('fc2/output',x_encoder2)
            tf.summary.histogram('fc2/weights',self.weights['fc2_w'])
            tf.summary.histogram('fc2/biases', self.biases['fc2_b'])
        with tf.name_scope('fc3'):
            self.x_encoder3 = self.fc_layer(x_encoder2, self.weights['fc3_w'], self.biases['fc3_b'], activation_function=tf.nn.tanh)
            tf.summary.histogram('fc3/output',self.x_encoder3)
            tf.summary.histogram('fc3/weights',self.weights['fc3_w'])
            tf.summary.histogram('fc3/biases', self.biases['fc3_b'])

        with tf.name_scope('regression1'):
            self.x_regression1 = self.fc_layer(self.x_encoder3, self.weights['regression1_w'], self.biases['regression1_b'], activation_function=tf.nn.tanh)
            tf.summary.histogram('regression1/output',self.x_regression1)
            tf.summary.histogram('regression1/weights',self.weights['regression1_w'])
            tf.summary.histogram('regression1/biases', self.biases['regression1_b'])

        with tf.name_scope('regression2'):
            self.x_regression2 = self.fc_layer(self.x_regression1, self.weights['regression2_w'], self.biases['regression2_b'], activation_function=None)
            tf.summary.histogram('regression2/output',self.x_regression2)
            tf.summary.histogram('regression2/weights',self.weights['regression2_w'])
            tf.summary.histogram('regression2/biases', self.biases['regression2_b'])

        with tf.name_scope('Regularization'):
            #auto_regularizers = tf.nn.l2_loss(self.weights['fc1_w']) + tf.nn.l2_loss(self.weights['fc2_w']) + tf.nn.l2_loss(self.weights['fc3_w'])
            regress_regularizers = tf.nn.l2_loss(self.weights['regression1_w']) + tf.nn.l2_loss(self.weights['regression2_w'])
            #regularizers = auto_regularizers + regress_regularizers
            regularizers = regress_regularizers
            tf.summary.scalar('regularization', regularizers)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.gt - self.x_regression2), 1), name='loss')
            #self.loss = tf.add(tf.reduce_mean(tf.reduce_sum(tf.square(self.gt - self.x_regression2), 1)), 0.1*regularizers, name='loss')
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def savemodel(self, path):
        model_data = {
            "weights": {
                'fc1_w': self.weights['fc1_w'].eval(),
                'fc2_w': self.weights['fc2_w'].eval(),
                'fc3_w': self.weights['fc3_w'].eval(),
                'regression1_w': self.weights['regression1_w'].eval(),
                'regression2_w': self.weights['regression2_w'].eval()
            },
            "biases": {
                'fc1_b': self.biases['fc1_b'].eval(),
                'fc2_b': self.biases['fc2_b'].eval(),
                'fc3_b': self.biases['fc3_b'].eval(),
                'regression1_b': self.biases['regression1_b'].eval(),
                'regression2_b': self.biases['regression2_b'].eval()
            }
        }
        np.save(path, model_data)
        print ("Save the model to %s" % path)