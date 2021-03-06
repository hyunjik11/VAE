import numpy as np
import tensorflow as tf
import os
import urllib
import matplotlib.pyplot as plt
#%matplotlib inline

np.random.seed(0)
tf.set_random_seed(0)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data")
n_train_samples = mnist.train.num_examples
n_valid_samples = mnist.validation.num_examples
n_test_samples = mnist.test.num_examples
# n_total_samples = n_train_samples + n_valid_samples
# using images with pixel values in {0,1}. i.e. p(x|z) is bernoulli

def lines_to_np_array(lines):
    return np.array([[int(i) for i in line.split()] for line in lines])

with open(os.path.join("MNIST_data", 'binarized_mnist_train.amat')) as f:
    lines = f.readlines()
    train_data = lines_to_np_array(lines).astype('float32')
with open(os.path.join("MNIST_data", 'binarized_mnist_valid.amat')) as f:
    lines = f.readlines()
    validation_data = lines_to_np_array(lines).astype('float32')
with open(os.path.join("MNIST_data", 'binarized_mnist_test.amat')) as f:
    lines = f.readlines()
    test_data = lines_to_np_array(lines).astype('float32')

train_data = np.vstack((train_data,validation_data))
n_train_samples = train_data.shape[0]
n_test_samples = test_data.shape[0]


def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)

def gauss_init(n_params,stddev=1): #initialise params by Gaussians
    return tf.random_normal([n_params],stddev=stddev,dtype=tf.float32)

def convert_col_idx(n_z,idx): 
# convert col indices of 2D array with n_z rows to indices for flattened vector
    ncol=len(idx)
    indices=np.empty(n_z*ncol,dtype=int)
    for i in xrange(0,ncol):
        indices[i*n_z:(i+1)*n_z]=np.arange(idx[i]*n_z,(idx[i]+1)*n_z)
    return indices

def bernoullisample(x):
    return np.random.binomial(1,x,size=x.shape)

class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, 
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # tf Graph input (placeholder objects can only be used as input, and can't be evaluated)
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        # placeholder object for indices of params in recog net
        # corresponding to  minibatch (use 0 indexing)
        self.idx = tf.placeholder(tf.int32, [batch_size*network_architecture["n_z"]])
        
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init) #run __init__ function
    
    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)
        # this feeds in the values of self.network_architecture as arguments to _initialize_weights
        
        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        n_z = self.network_architecture["n_z"]
        n_samples = self.network_architecture["n_samples"]
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["params_recog"],n_z,n_samples)

        # Draw one sample z from Gaussian distribution
        # n_z is dimensionality of latent space
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])
            
    def _initialize_weights(self, n_samples, 
                            n_hidden_gener_1, n_hidden_gener_2, n_input, n_z):
        # initialize weights, stored in dictionary all_weights (member of VAE class)
        # these are all the variables that will be learned
        # Use Xaiver init for weights, 0 for biases
        all_weights = dict()
        all_weights['params_recog'] = {
            'mu': tf.Variable(gauss_init(n_z*n_samples)),
            'log_sigma_sq': tf.Variable(gauss_init(n_z*n_samples))}
            # stack each column of params(n_z by n_samples) into one column vector
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights
            
    def _recognition_network(self, params, n_z, n_samples):
        # The approximate posterior distribution, which is a fully factorized Gaussian 
        # in the latent variables.
        # Output mean and log_sigma_sq of z_n for the minibatch given by self.idx
        z_mean = tf.reshape(tf.gather(params['mu'],self.idx),[-1,n_z])
        z_log_sigma_sq = tf.reshape(tf.gather(params['log_sigma_sq'],self.idx),[-1,n_z])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto mean param of Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), 
                                           biases['b1'])) 
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), 
                                           biases['b2'])) 
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                 biases['out_mean']))
        return x_reconstr_mean
            
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluatio of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)
        # reduce_sum(x,1) is summing the rows of x (i.e. summing across input dim)
        # * is elementwise multiplication
        
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        # Use ADAM optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads_and_vars=optimizer.compute_gradients(self.cost)
        grads_and_vars = [(g,v) for g,v in grads_and_vars if g is not None]
        cgv = [(tf.clip_by_value(gv[0],-1.,1.),gv[1]) for gv in grads_and_vars]
        self.optimizer = optimizer.apply_gradients(cgv)
        
    def partial_fit(self, X, myidx):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X, self.idx: myidx})
        return cost
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian(Bernoulli?) distribution
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.x: X})
    
    def test_cost(self, X):
        """ Return cost of mini-batch without further training. """
        return self.sess.run(self.cost,feed_dict={self.x: X})

def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5, transfer_fct=tf.nn.softplus):
    vae = VariationalAutoencoder(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size,transfer_fct=transfer_fct)
    # Training cycle
    np.random.seed(0)
    idx_train = np.arange(n_train_samples)
    idx_test = np.arange(n_test_samples)
    for epoch in range(training_epochs):
        avg_cost = 0.
        testcost=0.
        total_train_batch = int(n_train_samples / batch_size)
        # total_valid_batch = int(n_valid_samples / batch_size)
        total_test_batch = int(n_test_samples / batch_size)
        # total_batch = total_train_batch + total_valid_batch
        # Loop over all batches
        for i in range(total_train_batch):
            batch_idx = idx_train[i*batch_size:(i+1)*batch_size]
            batch_xs = train_data[batch_idx,:]
            #if i < total_train_batch:
            #batch_xs, _ , batch_idx = mnist.train.next_batch(batch_size)
            #else:
            #    batch_xs, _ = mnist.validation.next_batch(batch_size)
            # Fit training using batch data
            # batch_xs = bernoullisample(batch_xs)
            idx = convert_col_idx(network_architecture["n_z"],batch_idx) 
                
            cost = vae.partial_fit(batch_xs, idx)
            # Compute average training ELBO
            avg_cost += cost / n_train_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            #for i in range(total_test_batch):
            #    batch_xs, _ = mnist.test.next_batch(batch_size)
            #    batch_xs = bernoullisample(batch_xs)
            #    testcost += vae.test_cost(batch_xs) / n_test_samples * batch_size
                
            print "Epoch:", '%04d' % (epoch+1), \
                  "trainELBO=", "{:.9f}".format(avg_cost)
        # shuffle training and test data
        np.random.shuffle(idx_train)
        np.random.shuffle(idx_test)

    return vae
        
network_architecture = \
    dict(n_samples=n_train_samples, # number of training data points
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=784, # MNIST data input (img shape: 28*28)
         n_z=20)  # dimensionality of latent space
vae = train(network_architecture, batch_size=100,training_epochs=1000, display_step=10, transfer_fct=tf.nn.relu)
vae.sess.close()
