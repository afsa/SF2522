import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Dimension of the target f: R^d -> R
d = 2

# Number of training points
N = 1000

# Number of nodes in the hidden layer
K = 10

# Number of training steps
M = 40000

# Learning rate
dt = 0.005

# The total training time
T = M*dt

# Create a placeholder for the input
inp = tf.placeholder(dtype=tf.float32, shape=(None, d))

# Create mock training and testing data
x_training = np.random.uniform(-4, 4, size=(N, d))
x_test = np.random.uniform(-4, 4, size=(100, d))

# Create the mock target
target_fcn = tf.reshape(tf.square(tf.norm(inp - 0.5, axis=1)), (-1, 1))


def neural_network_model(inp):
    """
    Generate the neural network
    :param inp: the input tensor
    :return: the neural network tensor
    """
    initializer = tf.initializers.random_normal
    activation_fcn_1 = tf.nn.sigmoid
    activation_fcn_output = None
    num_output_nodes = 1
    
    hidden_layer_1 = tf.layers.dense(inputs=inp,
                                     units=K,
                                     activation=activation_fcn_1,
                                     kernel_initializer=initializer())
    
    output = tf.layers.dense(inputs=hidden_layer_1,
                             units=num_output_nodes,
                             activation=activation_fcn_output,
                             use_bias=False,
                             kernel_initializer=initializer())
    return output


def train_neural_network(sess):
    """
    Train a neural network with mean square loss
    :param sess: the session to use during training
    :return: the trained neural network and the expected errors
    """
    alpha = neural_network_model(inp)
    cost = tf.reduce_mean(tf.square(alpha - target_fcn))
    optimizer = tf.train.GradientDescentOptimizer(dt)
    train = optimizer.minimize(cost)
    E_1 = []
    
    sess.run(tf.global_variables_initializer())
    
    for m in range(0, M):
        n = np.random.randint(0, N)
        x_in = x_training[n:n + 1]
        sess.run(train, feed_dict={inp: x_in})
        
        if m % int(M/100) == 0:
            E_1.append(cost.eval(feed_dict={inp: x_test}))
        if m % int(M/10) == 0:
            print(m)
            
    return alpha, E_1


# Plot the target function and the neural network in a two dimensional plane.
# Create data points for the axis we wish to plot
one_d_pts = np.linspace(-4, 4, 300)

# Set all other coordinates to zero
pts = np.vstack([one_d_pts, np.zeros((d - 1, 300))]).T

# Run the training procedure and retrieve values of the target and the network
with tf.Session() as sess:
    alpha, E_1 = train_neural_network(sess)
    target_fcn_vals = target_fcn.eval(feed_dict={inp: pts})
    alpha_vals = alpha.eval(feed_dict={inp: pts})

# Plot the data
plt.figure('Learned function')
plt.plot(one_d_pts, target_fcn_vals, label='Target function')
plt.plot(one_d_pts, alpha_vals, label=r'$\alpha$')
plt.legend()
plt.show()

plt.figure('Test error')
plt.semilogy(np.linspace(0, T, len(E_1)), E_1, label='$E_1$')
plt.legend()
plt.show()
