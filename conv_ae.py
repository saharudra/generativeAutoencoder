'''
author = rudra saha (https://sharudra.github.io)
The batch_xs that will be the part of the separate training and testing file 
will take input from the mnist thingy and that will be reshaped.
'''
import tensorflow as tf
import numpy as np

class ConvolutionalAutoencoder:
    def __init__(self,
                 input_dim,  # Something that we define
                 batch_size=100,
                 activation_func=tf.nn.elu,
                 output_activation_func=tf.nn.sigmoid):

        self.graph = tf.Graph()
        self.activation_func = activation_func
        self.output_activation_func = output_activation_func
        self.input_dim = input_dim
        self.batch_size = batch_size

        with self.graph.as_default():
            # Input x variable
            self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
            self.z = tf.placeholder(tf.float32)
    '''
    One thing that I found that this might be one of the sources of bottleneck along with the 
    constant output after every epoch. Zero in on the one that truly is.
    '''
    # def conv2d(self, x, W, strides=[1, 1, 1, 1]):
    #     return tf.nn.conv2d(input=x, filter=W, strides=strides, padding='SAME')
    #
    # def max_pool_2x2(self, x):
    #     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def encoder(self, x, reuse=False):
        '''
        The encoder_decoder function
        Takes in the image
        Ouputs the hidden vector and the image
        '''
        if (reuse):
            tf.get_variable_scope().reuse_variables()
        with self.graph.as_default():
            # Encoder starts from here
            # First convolutional layer with elu activation
            W_conv1 = tf.get_variable('en_wconv1', [5, 5, 1, 8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv1 = tf.get_variable('en_bconv1', [8], initializer=tf.constant_initializer(0))
            h_conv1 = tf.nn.elu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

            # First batch normalization and pooling layer
            norm1 = tf.nn.lrn(h_conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
            h_pool1 = tf.nn.max_pool(norm1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            # Second convolutional layer with elu activation
            W_conv2 = tf.get_variable('en_wconv2', [5, 5, 8, 16],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv2 = tf.get_variable('en_bconv2', [16], initializer=tf.constant_initializer(0))
            h_conv2 = tf.nn.elu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

            # Second batch normalization and pooling layer
            norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
            h_pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            '''
            The below code has to be tweaked for proper number of dimensions for the mnist dataset.
            Post the application of convolutions, the assumed outcome is of 7 * 7 * 16 in order to 
            restore the number of dimensions to 784. Don't think this will be a good idea to go bigger
            with 4 * 4 * 32
            '''
            # # Third convolutional layer with elu activation, will try relu later as well
            # W_conv3 = tf.get_variable('en_conv3', [3, 3, 16, 32],
            #                           initializer=tf.truncated_normal_initializer(stddev=0.02))
            # b_conv3 = tf.get_variable('en_bconv3', [32], initializer=tf.constant_initializer(0))
            # h_conv3 = self.activation_func(self.conv2d(h_pool2, W_conv3) + b_conv3)

            # Another normalization layer, see it's viability and check if removing this is improving anything
            # norm2 = tf.nn.lrn(h_conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

            # First fully connected layer of dimensions [784, 100]
            W_fc1 = tf.get_variable('d_wfc1', [7 * 7 * 16, 100],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_fc1 = tf.get_variable('d_bfc1', [100], initializer=tf.constant_initializer(0))
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 16])

            # The following is the autoencoder output of dimensionality [None, 100]
            # The Decoder starts from here
            self.z = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            print("Dimension of z i.e. our latent vector is " + str(self.z.get_shape().as_list()))
        return self.z

    def decoder(self, reuse= False):

        if(reuse):
            tf.get_variable_scope().reuse_variables()
        with self.graph.as_default():
            g_dim = 64  # Number of filters of first layer of decoder
            c_dim = 1  # Color dimension of output (MNIST is grayscale, so c_dim = 1 for us)
            s = 28  # Output size of the image
            s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(
                s / 16)  # We want to slowly upscale the image, so these values will help
            # make that change gradual as a result of which the output image will be better.

            h0 = tf.reshape(self.z, [self.batch_size, s16 + 1, s16 + 1, 25])
            h0 = tf.nn.relu(h0)
            # Dimensions of h0 = batch_size x 2 x 2 x 25

            # First DeConv Layer
            output1_shape = [self.batch_size, s8, s8, g_dim * 4]
            W_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1], int(h0.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))
            H_conv1 = tf.nn.conv2d_transpose(h0, W_conv1, output_shape=output1_shape, strides=[1, 2, 2, 1], padding='SAME')
            H_conv1 = tf.contrib.layers.batch_norm(inputs=H_conv1, center=True, scale=True, is_training=True, scope="g_bn1")
            H_conv1 = tf.nn.relu(H_conv1)
            # Dimensions of H_conv1 = batch_size x 3 x 3 x 256

            # Second DeConv Layer
            output2_shape = [self.batch_size, s4 - 1, s4 - 1, g_dim * 2]
            W_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(H_conv1.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
            H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape, strides=[1, 2, 2, 1],
                                             padding='SAME')
            H_conv2 = tf.contrib.layers.batch_norm(inputs=H_conv2, center=True, scale=True, is_training=True, scope="g_bn2")
            H_conv2 = tf.nn.relu(H_conv2)
            # Dimensions of H_conv2 = batch_size x 6 x 6 x 128

            # Third DeConv Layer
            output3_shape = [self.batch_size, s2 - 2, s2 - 2, g_dim * 1]
            W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
            H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, strides=[1, 2, 2, 1],
                                             padding='SAME')
            H_conv3 = tf.contrib.layers.batch_norm(inputs=H_conv3, center=True, scale=True, is_training=True, scope="g_bn3")
            H_conv3 = tf.nn.relu(H_conv3)
            # Dimensions of H_conv3 = batch_size x 12 x 12 x 64

            # Fourth DeConv Layer
            output4_shape = [self.batch_size, s, s, c_dim]
            W_conv4 = tf.get_variable('g_wconv4', [5, 5, output4_shape[-1], int(H_conv3.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
            H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape, strides=[1, 2, 2, 1],
                                             padding='VALID')
            H_conv4 = tf.nn.tanh(H_conv4)

            print("Dimension of the output of the decoder is " + str(H_conv4.get_shape().as_list()))

            # Dimensions of H_conv4 = batch_size x 28 x 28 x 1
        return H_conv4

    def reconstruction_loss(self, reuse=False):
        with self.graph.as_default():
            print("getting into solving the reconstruction loss")
            z_vector = self.encoder(self.x, reuse)
            self.z = z_vector
            y = self.decoder(reuse)
            cost = tf.reduce_sum(tf.square(y - self.x))
            output = [cost, y]
            return cost
