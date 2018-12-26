import tensorflow as tf
import numpy as np


class All:
    def __init__(self, model_path):

        input_nodes = 5
        label_nodes = 2
        ## input layer ##
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.float32, [None, input_nodes], name='x')
            tf.summary.histogram('input/x', self.x)

        if model_path == None:
            input_nodes = 5
            layer1_nodes = 15
            layer2_nodes = 10
            layer3_nodes = 5

            self.weights = {
                'encoder1_w': self.weight([input_nodes, layer1_nodes], 'encoder1'),
                'encoder2_w': self.weight([layer1_nodes, layer2_nodes], 'encoder2'),
                'encoder3_w': self.weight([layer2_nodes, layer3_nodes], 'encoder3')
            }

            self.biases = {
                'encoder1_b': self.bias([layer1_nodes], 'encoder1'),
                'encoder2_b': self.bias([layer2_nodes], 'encoder2'),
                'encoder3_b': self.bias([layer3_nodes], 'encoder3'),
                'decoder1_b': self.bias([layer2_nodes], 'decoder1'),
                'decoder2_b': self.bias([layer1_nodes], 'decoder2'),
                'decoder3_b': self.bias([input_nodes], 'decoder3')
            }

        if model_path != None:

            restore_model = np.load(model_path).item()

            self.weights = {
                'encoder1_w': tf.Variable(tf.constant(restore_model["weights"]["encoder1_w"]), name='encoder1_w'),
                'encoder2_w': tf.Variable(tf.constant(restore_model["weights"]["encoder2_w"]), name='encoder2_w'),
                'encoder3_w': tf.Variable(tf.constant(restore_model["weights"]["encoder3_w"]), name='encoder3_w'),
            }
            self.biases = {

                'encoder1_b': tf.Variable(tf.constant(restore_model["biases"]["encoder1_b"]), name='encoder1_b'),
                'encoder2_b': tf.Variable(tf.constant(restore_model["biases"]["encoder2_b"]), name='encoder2_b'),
                'encoder3_b': tf.Variable(tf.constant(restore_model["biases"]["encoder3_b"]), name='encoder3_b'),
                'decoder1_b': tf.Variable(tf.constant(restore_model["biases"]["decoder1_b"]), name='decoder1_b'),
                'decoder2_b': tf.Variable(tf.constant(restore_model["biases"]["decoder2_b"]), name='decoder2_b'),
                'decoder3_b': tf.Variable(tf.constant(restore_model["biases"]["decoder3_b"]), name='decoder3_b')
            }

    def weight(self,shape,layername):
        W = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.3, dtype = tf.float32), name=layername+'_w')
        return W

    def bias(self,shape,layername):
        B = tf.Variable(tf.constant(0.3, shape=shape, dtype = tf.float32), name=layername+'_b')
        return B

    def initial(self):
        #init = tf.initialize_all_variables()
        init= tf.global_variables_initializer()
        return init

    def fc_layer(self,inputs,w,b,activation_function=tf.nn.tanh):
        Wx_plus_b = tf.nn.bias_add(tf.matmul(inputs,w),b)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    def build(self):

        with tf.name_scope('encoder1'):
            self.x_encoder1 = self.fc_layer(self.x, self.weights['encoder1_w'], self.biases['encoder1_b'], activation_function=tf.nn.tanh)
            tf.summary.histogram('encoder1/output',self.x_encoder1)
            tf.summary.histogram('encoder1/weights',self.weights['encoder1_w'])
            tf.summary.histogram('encoder1/biases', self.biases['encoder1_b'])

        with tf.name_scope('encoder2'):
            self.x_encoder2 = self.fc_layer(self.x_encoder1, self.weights['encoder2_w'], self.biases['encoder2_b'], activation_function=tf.nn.tanh)
            tf.summary.histogram('encoder2/output',self.x_encoder2)
            tf.summary.histogram('encoder2/weights',self.weights['encoder2_w'])
            tf.summary.histogram('encoder2/biases', self.biases['encoder2_b'])

        with tf.name_scope('encoder3'):
            self.x_encoder3 = self.fc_layer(self.x_encoder2, self.weights['encoder3_w'], self.biases['encoder3_b'], activation_function=tf.nn.tanh)
            tf.summary.histogram('encoder3/output',self.x_encoder3)
            tf.summary.histogram('encoder3/weights',self.weights['encoder3_w'])
            tf.summary.histogram('encoder3/biases', self.biases['encoder3_b'])

        with tf.name_scope('decoder1'):
            self.x_decoder1 = self.fc_layer(self.x_encoder3, tf.transpose(self.weights['encoder3_w']), self.biases['decoder1_b'], activation_function=tf.nn.tanh)
            tf.summary.histogram('decoder1/output',self.x_decoder1)
            tf.summary.histogram('decoder1/weights',tf.transpose(self.weights['encoder3_w']))
            tf.summary.histogram('decoder1/biases', self.biases['decoder1_b'])

        with tf.name_scope('decoder2'):
            self.x_decoder2 = self.fc_layer(self.x_decoder1, tf.transpose(self.weights['encoder2_w']), self.biases['decoder2_b'], activation_function=tf.nn.tanh)
            tf.summary.histogram('decoder2/output',self.x_decoder2)
            tf.summary.histogram('decoder2/weights',tf.transpose(self.weights['encoder2_w']))
            tf.summary.histogram('decoder2/biases', self.biases['decoder2_b'])

        with tf.name_scope('decoder3'):
            self.x_decoder3 = self.fc_layer(self.x_decoder2, tf.transpose(self.weights['encoder1_w']), self.biases['decoder3_b'], activation_function=tf.nn.tanh)
            tf.summary.histogram('decoder3/output',self.x_decoder3)
            tf.summary.histogram('decoder3/weights',tf.transpose(self.weights['encoder1_w']))
            tf.summary.histogram('decoder3/biases', self.biases['decoder3_b'])

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.x_decoder3 - self.x),1), name='loss')
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def savemodel(self, path):
        model_data = {
            "weights": {
                'encoder1_w': self.weights['encoder1_w'].eval(),
                'encoder2_w': self.weights['encoder2_w'].eval(),
                'encoder3_w': self.weights['encoder3_w'].eval()
            },
            "biases": {
                'encoder1_b': self.biases['encoder1_b'].eval(),
                'encoder2_b': self.biases['encoder2_b'].eval(),
                'encoder3_b': self.biases['encoder3_b'].eval()
            }
        }
        np.save(path, model_data)
        print ("Save the model to %s" % path)