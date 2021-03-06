import numpy as np
import tensorflow as tf
import os
import urllib
import matplotlib.pyplot as plt
#%matplotlib inline

np.random.seed(0)
tf.set_random_seed(0)
"""
# import MNIST data - scripts were installed with tensorflow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data")
n_train_samples = mnist.train.num_examples
n_test_samples = mnist.test.num_examples
# using images with pixel values in {0,1}. i.e. p(x|z) is bernoulli
"""

# download fixed binarized mnist dataset
"""
    subdatasets = ['train', 'valid', 'test']
    for subdataset in subdatasets:
    filename = 'binarized_mnist_{}.amat'.format(subdataset)
    url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(subdataset)
    local_filename = os.path.join("MNIST_data", filename)
    urllib.urlretrieve(url, local_filename)
    """
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
    return tf.random_uniform((fan_in, fan_out), \
                             minval=low, maxval=high, \
                             dtype=tf.float32)

def bernoullisample(x):
    return np.random.binomial(1,x,size=x.shape)

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'): # name_scope used to visualise variables
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        # tf.histogram_summary(name, var)

class VariationalAutoencoder(object):
    """
        Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow on binarised MNIST data.
        This implementation uses an isotropic Gaussian prior on the latent variables z
        and Bernoulli likelihood for p(x|z)
        See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
        """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        """
        tf creates a computational graph, with nodes being tf variables and operations
        The tf graph consists of tf variables - think of as symbolic variables (tensors) that are only evaluated when explicitly told so
        They are tf operations applied to other tf variables / placeholders
        These relations make up the edges of the tf graph
        """
        
        # tf Graph input (placeholder objects can only be used as input, and can't be evaluated)
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        # Create autoencoder network
        self._create_network()
        
        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Launch the session
        self.sess = tf.InteractiveSession()
        
        # Merge summaries
        self.merged = tf.merge_all_summaries()
        
        # Call AFTER defining all tf variables
        init = tf.initialize_all_variables()
        self.sess.run(init)
    
    def _create_network(self):
        # Initialize autoencoder network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)
        # network_architecture is a dictionary of the dimensions of each layer in generator and recognition network
        # network_weights is a dictionary of tensors
        
        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"],
                                      network_weights["biases_recog"])
        # now z_mean, z_log_sigma_sq are attributes of VAE class that are tf variables
        
        # Draw one sample z from Gaussian distribution
        # n_z is dimensionality of latent space
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((tf.shape(self.x)[0], n_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, \
                       tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
       
        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
           self._generator_network(network_weights["weights_gener"], \
                                   network_weights["biases_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, \
                            n_hidden_gener_1,  n_hidden_gener_2, \
                            n_input, n_z):
        # initialize weights, stored in dictionary all_weights (member of VAE class)
        # these are all the tf variables that will be learned
        # Use Xaiver init for weights, 0 for biases
        all_weights = dict()
        all_weights['weights_recog'] = {
                'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
                'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
                'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
                'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
                'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
                'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
                'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
                'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
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
        variable_summaries(all_weights['weights_recog']['h1'],'recog_h1')
        variable_summaries(all_weights['weights_recog']['h2'],'recog_h2')
        variable_summaries(all_weights['weights_recog']['out_mean'],'recog_weights_out_mean')
        variable_summaries(all_weights['weights_recog']['out_log_sigma'],'recog_weights_out_log_sigma')
        variable_summaries(all_weights['biases_recog']['b1'],'recog_b1')
        variable_summaries(all_weights['biases_recog']['b2'],'recog_b2')
        variable_summaries(all_weights['biases_recog']['out_mean'],'recog_biases_out_mean')
        variable_summaries(all_weights['biases_recog']['out_log_sigma'],'recog_biases_out_log_sigma')
        variable_summaries(all_weights['weights_gener']['h1'],'gener_h1')
        variable_summaries(all_weights['weights_gener']['h2'],'gener_h2')
        variable_summaries(all_weights['weights_gener']['out_mean'],'gener_weights_out_mean')
        variable_summaries(all_weights['weights_gener']['out_log_sigma'],'gener_weights_out_log_sigma')
        variable_summaries(all_weights['biases_gener']['b1'],'gener_b1')
        variable_summaries(all_weights['biases_gener']['b2'],'gener_b2')
        variable_summaries(all_weights['biases_gener']['out_mean'],'gener_biases_out_mean')
        variable_summaries(all_weights['biases_gener']['out_log_sigma'],'gener_biases_out_log_sigma')
        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto mean and log_sigma_sq vector of a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),biases['out_mean'])
        z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']),biases['out_log_sigma'])
        variable_summaries(layer_1,'recog_layer_1')
        variable_summaries(layer_2,'recog_layer_2')
        variable_summaries(z_mean,'recog_z_mean')
        variable_summaries(z_mean,'recog_z_log_sigma_sq')
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto mean param of Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),biases['b2']))
        x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),biases['out_mean']))
        variable_summaries(layer_1,'gener_layer_1')
        variable_summaries(layer_2,'gener_layer_2')
        variable_summaries(x_reconstr_mean,'gener_z_mean')
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
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean) + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),1)
        # reduce_sum(x,1) is summing the rows of x (i.e. summing across input dim)
        # * is elementwise multiplication
       
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior. (Distance between p(z_n) and q(z_n|x_n) )
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                          - tf.square(self.z_mean)
                                          - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        variable_summaries(latent_loss,'latent_loss')
        variable_summaries(reconstr_loss,'reconstr_loss')
        # Use ADAM optimizer
        self.optimizer = \
           tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
            
            Return cost of mini-batch.
            """
        # Run one step of optimizer then evaluate new cost
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X})
        return cost

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
            # Note: This maps to mean of distribution, we could alternatively
            # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
        
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
    train_writer = tf.train.SummaryWriter('/homes/hkim/VAE/summaries/train',vae.sess.graph)
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
            batch_xs = train_data[idx_train[i*batch_size:(i+1)*batch_size],:]
                    
            cost = vae.partial_fit(batch_xs)
            # Compute average training ELBO
            avg_cost += cost / n_train_samples * batch_size
                        
            # Display logs per epoch step
        if epoch % display_step == 0:
            for i in range(total_test_batch):
                batch_xs = test_data[idx_test[i*batch_size:(i+1)*batch_size],:]
                testcost += vae.test_cost(batch_xs) / n_test_samples * batch_size
            temp = vae.sess.run(vae.cost,feed_dict={vae.x : train_data})
            summary = vae.sess.run(vae.merged,feed_dict={vae.x : train_data})
            train_writer.add_summary(summary,epoch)
            print "Epoch:", '%04d' % (epoch+1), \
                  "trainELBO=", "{:.9f}".format(avg_cost), \
                  "testELBO=", "{:.9f}".format(testcost)
        # shuffle training and test data
        np.random.shuffle(idx_train)
        np.random.shuffle(idx_test)
    train_writer.close()
    return vae

network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=784, # MNIST data input (img shape: 28*28)
         n_z=20)  # dimensionality of latent space

#with tf.device('/gpu:0'): (this is done by default on gpu machines)
vae = train(network_architecture, training_epochs=200, display_step=1, transfer_fct=tf.nn.relu)
vae.sess.close() # get errors when starting a new session without closing an old one

