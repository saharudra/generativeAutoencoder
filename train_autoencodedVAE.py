'''
author = rudra saha

the batch_xs should be reshaped to [None, 28, 28, 1]
'''

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import time
import pandas as pd
import os

from save_zees import create_save_z
from conv_ae import ConvolutionalAutoencoder
results_dir = '/home/exx/Rudra/autoencodedVAE/results'
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def train_mnist_conv_vae(epochs = 1500,
                              batch_size = 100,
                              optimizer = 'adam',
                              test_full = False):
    # Getting input z variables in the form of numpy matrix
    num_batches = 549

    input_dim = 784
    learning_rate = 1e-4

    conv_vae = ConvolutionalAutoencoder(
        input_dim=input_dim,
        batch_size=batch_size,
    )

    with conv_vae.graph.as_default():
        if optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate).minimize(conv_vae.reconstruction_loss())
        else:
            raise ValueError("Optimizer '{}' is not supported".format(optimizer))

        with tf.Session(graph=conv_vae.graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            saver.restore(sess, '/home/exx/Rudra/autoencodedVAE/checkpoint_autoencoder_model.ckpt')
            print("Model restored")

#             for epoch_i in range(epochs):
#                 print("In epoch " + str(epoch_i))
#                 for batch_i in range(mnist.train.num_examples // batch_size):
#                     batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#                     batch_xs = np.array(batch_xs)
#                     print(batch_xs.shape)
#                     batch_xs = batch_xs.reshape((batch_size, 28, 28, 1))
#                     sess.run(opt, feed_dict={conv_vae.x: batch_xs})
#                 print(epoch_i)
#                 if (epoch_i % 10 == 0):
#                     epoch_10_time = time.time()
#                     time_10 = epoch_10_time - start_time
#                     print("Time elapsed till " + str(epoch_i) + " : " + str(time_10))
#                     print(epoch_i, sess.run(conv_vae.reconstruction_loss(reuse=True), feed_dict={conv_vae.x: batch_xs}))
#
#                     # Checking the output and the shape of z in order to save them properly
#                     # Saving it in a csv file to see how it does right now
#                     # The z output is 100 x 100 i.e. 100 elements of z for each of the example
#                     # in the current batch
#                     # z, recon = sess.run(conv_vae.encoder_decoder(batch_xs, reuse=True), feed_dict={conv_vae.x: batch_xs})
#                     # print(z)
#                     # print(np.shape(z))
#                     # f_handle = open('z_file.csv', 'wb')
#                     # np.savetxt(f_handle, z, fmt='%f')
#                     # f_handle.close()
#
# #            saver.save(sess, 'checkpoint_autoencoder_model.ckpt')
#             create_save_z('mnist' ,mnist, 'mnist_z_from_ae_file.csv', batch_size, conv_vae, sess)
            # save_path = saver.save(sess, 'model_conv_ae.cpkt')
            # print("Model saved in file: %s" % save_path)
            # '''
            # Visualization code
            # '''
            n_examples = 100
            test_xs, test_ys = mnist.test.next_batch(n_examples)
            test_xs = test_xs.reshape((n_examples, 28, 28, 1))
            # z = sess.run(conv_vae.encoder(test_xs, reuse=True), feed_dict={conv_vae.x: test_xs})
            # reconstructed_images = sess.run(conv_vae.decoder(reuse=True), feed_dict={conv_vae.z: z})
            #
            # fig, axs = plt.subplots(2, n_examples, figsize=(100, 10))
            #
            # for example_i in range(n_examples):
            #     axs[0][example_i].imshow(
            #         np.reshape(test_xs[example_i, :], (28, 28)))
            #     axs[1][example_i].imshow(
            #         np.reshape(
            #             np.reshape(reconstructed_images[example_i, ...], (784,)),
            #             (28, 28)))
            #     if (example_i == n_examples-1):
            #         plt.savefig("ae_reconstructed_for_check_fig.pdf")

            # shoot_up_z = batch_matrix_vae[0]
            # _, reconstructed_images_vae = sess.run(conv_vae.encoder_decoder())
            # fig.show()
            # plt.draw()
            # plt.waitforbuttonpress()

            '''
            Visualization of latent space just being shot up in the decoder of the AE.
            The current saved file will be used again and turned into a reconstructed image.
            '''
            file_z_ae = pd.read_csv('mnist_z_from_ae_file.csv')
            file_z_ae = file_z_ae.as_matrix()[:100, :]
            print("part 1")
            reconstruction_by_shooting_up_just_ae = sess.run(conv_vae.decoder(reuse= True), feed_dict={conv_vae.z: file_z_ae})
            fig, axs = plt.subplots(2, n_examples, figsize=(100, 10))

            for example_i in range(n_examples):
                axs[1][example_i].imshow(
                    np.reshape(test_xs[example_i, :], (28, 28)))
                axs[0][example_i].imshow(
                    np.reshape(
                        np.reshape(reconstruction_by_shooting_up_just_ae[example_i, ...], (784,)),
                        (28, 28)))
                if (example_i == n_examples - 1):
                    plt.savefig("image_from_decoder_checkpoint_restore_check.pdf")


            '''
            Shooting up code
            '''
            print("part_2")
            file = pd.read_csv('/home/exx/Rudra/autoencodedVAE/latent_values_from_iwae_layers2_latent_25_10_minibatch_10_lr_1e-4.csv')
            file = file.as_matrix()
            print(file.shape)
            print(file[0])
            print(file[0][0])
            reconstructed_images_vae = sess.run(conv_vae.decoder(reuse= True), feed_dict= {conv_vae.z: file})

            fig, axs = plt.subplots(2, n_examples, figsize=(100, 10))

            for example_i in range(n_examples):
                axs[0][example_i].imshow(
                    np.reshape(test_xs[example_i, :], (28, 28)))
                axs[1][example_i].imshow(
                    np.reshape(
                        np.reshape(reconstructed_images_vae[example_i, ...], (784,)),
                        (28, 28)))
                if (example_i == n_examples - 1):
                    image_name = 'From_IWAE_gaussian_latent_layers2_25_10_mini_10.pdf'
                    plt.savefig(os.path.join(results_dir, image_name))


if __name__ == '__main__':
    start_time = time.time()
    train_mnist_conv_vae()
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken to run convoluted autoencoder: " + str(total_time))
