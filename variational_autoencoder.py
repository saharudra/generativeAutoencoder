'''
author: rudra saha (rudra.saha@asu.edu)
Code based on https://jmetzen.github.io/2015-11-27/vae.html
Complete the TODOs
'''
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import utils
import theano
import theano.tensor as T
import collections
from utils import t_repeat, log_mean_exp, reshape_and_tile_images

# log2pi = T.constant(np.log(2*np.pi).astype(theano.config.floatX)) # Constant tensor definition
# floatX = theano.config.floatX

np.random.seed(200)
tf.set_random_seed(0)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
num_samples = mnist.train.num_examples

def xavier_initialization(fan_in, fan_out, constant=1):
    '''
    Xavier initialization for weights
    :param fan_in: num of input connections to the current neuron
    :param fan_out: num of output connections from the current neuron
    :param constant: multiplicative factor
    :return: a tensor with the required initialization
    '''
    low = -constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))

    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

'''
TODO::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Add log likelihood estimation for the vae for comparison.
Cross entropy loss with gaussian
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
'''

class VariationalAutoencoder(object):
    '''
    Convolutional variational autoencoder.
    The idea is to structure the latent space so that while sampling from it we can
    define the region of the latent space that will result in a particular type of
    output.
    For example, say we are working with mnist. Now while sampling from the latent
    space, we would want the output to be 0 when we are sampling from near the origin
    of the latent space and as we gradually go outwards from the origin, we expect
    this to produce the various mnist classes.
    Essentially think of this as a 100 dimensional sphere and we are sampling from
    this and at each of the level, the decoder will generate an output depending
    upon the sampling points distance from the origin.

    max Lvae = E[p(x|z)] - KL[q(z|x)||p(z)] - lambda*E[f(decoded(z)) - ||z||]

    Here E[p(x|z)] is the reconstruction loss of the vae, the KL divergence is
    between a posterior distribution and some prior distribution which is usually
    taken as a gaussian distribution and we use the reparameterization trick to sample
    from such a posterior. f() is a known function that takes the generated output
    of the decoder and maps it to a particular discrete or continous thing. For our
    this will be the class label. \\z\\ is the L_2 distance of the point from which
    we are currently sampling with the origin i.e. a vector of all 0s.

    The last term can be considered similar to a regularization function that will enforce
    that while sampling, say we generated a 0 from some point in the latent space, this
    point should be as near to the origin as possible, similarly if we generated a 7,
    we want it to be atleast a distance of 7 away from the origin for this to contribute
    less towards the loss function.

    This predefined function (currently which is just the class label) can be a neural
    network that takes in an input and outputs some predefined class.
    This can be used for drug discovery problems which is our initial motivation in the
    first place.
    '''

    def __init__(self, network_arch, non_linearity=tf.nn.relu,
                 learning_rate=0.01, batch_size=100):
        with tf.device('/gpu:2'):
            self.network_architecture = network_architecture
            self.transfer_fct = non_linearity
            self.non_linearity = non_linearity
            self.learning_rate = learning_rate
            self.batch_size = batch_size

            # input to the training graph
            # network_arch is our networks architecture. n_input is the input node
            # to the network and x is being fed to it.
            self.x = tf.placeholder(tf.float32, [None, network_architecture['n_input']])

            # After defining what is being fed to the network, we have to create the network
            # architecture and it would be better to have an abstraction for this
            self.create_network()

            # Now we should define our loss function. It is being said that KL divergence
            # might not work, need to ask as to why this won't work and understand it for future
            # use.
            '''
            TODO: Understand why KL divergence won't work in this case, as we can scale the
            latent space as well using some transformation function and make the demarcations
            more clear.

            SOL:
            So KL divergence tries to match the p(z|x) to q(z) where we define q(z) as a gaussian
            distribution. Our regularization term will try to smoothen everything out in the
            form of uniform distribution. But while sampling in the next pass, we are again sampling
            from a gaussian distribution to match the KL divergence term.

            TODO::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            Search into spherical distribution and how that can be used.
            ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

            Sampling from latent space in the vae domain will then be equivalent to sampling
            from this transformed domain where the boundaries between the labels are further
            apart from each other.
            '''
            self.create_loss_optimizer()

            # initialize variables of the network
            init = tf.global_variables_initializer()

            # create a session
            self.sess = tf.InteractiveSession()
            self.sess.run(init)

    def create_network(self):
        # Initializing weights and biases, if convolution is required,
        # set them accordingly in the initialize_weights function and pass them
        # here.
        network_weights = self.initialize_weights(**self.network_architecture)

        # Using the encoder to determine the mean and the log variance of the
        # gaussian distribution in latent space, in the initialize_weights
        # method, we have separated the weights for the encoder and the
        # decoder networks along with biases.

        # The recogntion_network is synonymous to the encoder of the vae.
        self.z_mean, self.z_log_sigma_sq = self.recognition_network(network_weights["weights_recog"],
                                                                    network_weights["biases_recog"])
        # Drawing samples from a gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32)

        # z = mu + sigma * eps
        # This is the reparameterization trick for a gaussian prior.
        '''
        TODO:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        See how we can make this an N-dimensional spherical distribution and learn the
        radius of it as well. This might alleviate the problem with the kl divergence
        with a gaussian distribution and help us directly use vae to solve the problem
        instead of having an autoencoder to generate the latent space.
        :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        '''
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use the generator (synonymous to decoder) network to determine the mean of
        # Gaussian distribution of the reconstructed input.
        self.x_generated, self.x_reconstr_mean, self.x_reconstr_log_sigma_sq = \
            self.generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])

    def initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_initialization(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_initialization(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_initialization(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_initialization(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_initialization(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_initialization(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_initialization(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_initialization(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    def recognition_network(self, weights, biases):
        # The recognition network is synonymous to the encoder of the vae
        # We call this a probabilistic encoder as we have parameterized the
        # the latent space as a gaussian and then used the reparameterization
        # trick to sample from it.
        # This sampling is done in the create_network method.
        # We can have this as a shperical thing as well.

        # We will have to make changes here to make it convolutional as this
        # currently is a fully connected network.

        # See how we have called the weights and biases from the dictionary
        # and used the same name for them.
        '''
        TODO:
        Formulate the spherical thing later.
        '''
        layer_1 = self.non_linearity(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.non_linearity(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def generator_network(self, weights, biases):
        # The generator network is synonymous to the decoder of a vae. This maps a point
        # in the latent space onto a Bernoulli distribution in data space. The transformation
        # is parameterized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
                                 biases['out_mean']))
        x_reconstr_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma'])
        print("Shape of generator's mu and log_sigma_sq")
        print(np.shape(x_reconstr_mean), np.shape(x_reconstr_log_sigma_sq))
        print(x_reconstr_mean.get_shape().as_list(), x_reconstr_log_sigma_sq.get_shape().as_list())
        return x_reconstr_mean, x_reconstr_log_sigma_sq

    def create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        # This reconstruction loss defined here is cross-entropy loss
        # And in my current intuition is the reason why we are getting
        # a bernoulli distribution.
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)
        '''
        We can change this reconstruction loss evaluation to a direct difference
        between what is being reconstructed and the real image and that will
        make the output the way we desire (a 3 channel output), we can also have
        a deconcoluted decoder so that we have a 3-channel output towards the end.
        '''
        # reconstr_loss = tf.reduce_sum(tf.square(self.x - self.x_reconstr_mean))
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)  # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X})
        return cost

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        '''
        Calculation of self.z_mean is done inside the create_network() method
        by calling the recognition_network i.e. our encoder.
        '''
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=(self.batch_size, self.network_architecture["n_z"]))
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X})

    def measure_marginal_log_likelihood(self, dataset, subdataset, batch_size, num_samples,
                                        seed=123, z_mu=None):
        '''
        method used from iwae code, name of the method to be used to keep track of
        what is being implemented.

        The method in iwae takes in:

        model: we have direct access to model here as this method is being made part
        of the same class

        dataset: the input dataset

        subdataset: this is to identify whether it is training dataset or testing one,
        and we will do it accordingly.

        batch_size: this will help us determine the num of batches that has to be used

        num_samples
        '''
        print("Measuring {} log likelihood".format(subdataset))
        srng = utils.srng(seed)

        if subdataset == 'train':
            n_examples = mnist.train.num_examples
        else:
            n_examples = mnist.test.num_examples

        if n_examples % batch_size == 0:
            num_minibatches = n_examples // batch_size
        else:
            num_minibatches = n_examples // batch_size + 1

        index = T.lscalar('i')

        minibatch = dataset.minibatchIindex_minibatch_size(index, batch_size, subdataset= subdataset,
                                                           srng= srng)
        log_marginal_likelihood_estimate = self.log_marginal_likelihood_estimate(minibatch,
                                                                                 num_samples,srng)
        get_log_marginal_likelihood = theano.function([index], T.sum(log_marginal_likelihood_estimate))

        sum_of_log_likelihoods = 0.
        for i in range(num_minibatches):
            summand = get_log_marginal_likelihood(i)
            sum_of_log_likelihoods += summand

        marginal_log_likelihood = sum_of_log_likelihoods / n_examples

        return marginal_log_likelihood

    def log_marginal_likelihood_estimate(self, minibatch, num_samples, srng, z_mu=None):
        if z_mu is None:
            z_mu = np.random.normal(size=(self.batch_size, self.network_architecture["n_z"]))
        

        pass




'''
Function being used to train the vae.
'''
def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(num_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / num_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))
    return vae


network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=784, # MNIST data input (img shape: 28*28)
         n_z=20)  # dimensionality of latent space

vae = train(network_architecture, training_epochs=30)

x_sample = mnist.test.next_batch(100)[0]
x_reconstruct = vae.reconstruct(x_sample)

'''
Generating a sample
'''
x_generate = vae.generate()
print(np.shape(x_generate))
plt.figure(figsize=(8, 12))
for i in range(5):
    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(x_generate[i].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Generated")
    plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 12))
for i in range(5):
    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()
plt.show()
