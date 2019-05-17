from __future__ import print_function
import numpy as np
import tensorflow as tf


# Parameters
learning_rate = 1
epochs = 200
batch_size = 128
display_step = 100

n_hidden = 2 # 1st layer number of neurons
num_input = 2 
num_classes = 2 

input_data = np.array([[1,1],[-1,-1],[1,-1],[-1,1]])
y_cal = np.array([[1,0],[1,0],[0,1],[0,1]])
y = [1,1,0,0]
# 1,0,1,1,1,0,0
x_test = np.array([[0.9,0.9],[-0.9,0.8],[-0.8,-0.99],[1.2,1.5],[-1.3,-2],[0.3,0.1],[0,0]])
y_test = np.array([1,0,1,1,1,0,0])


# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h': tf.Variable(tf.random_normal([num_input, n_hidden],mean = 0, stddev = 0.02, dtype = tf.dtypes.float32, seed = 200)),
    'w_out': tf.Variable(tf.random_normal([n_hidden, num_classes], mean = 0, stddev = 0.02, dtype = tf.dtypes.float32, seed = 200))
}
biases = {
    'b': tf.zeros([n_hidden], dtype = tf.dtypes.float32),
    'b_out': tf.zeros([num_classes], dtype = tf.dtypes.float32)
}


# Create model
def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h']), biases['b'])
    layer_1 = tf.math.sigmoid(layer_1)
    out_layer = tf.add(tf.matmul(layer_1, weights['w_out']),biases['b_out'])
    return tf.math.sigmoid(out_layer)

logits = neural_net(X)

mse = tf.losses.mean_squared_error(y, logits)

# Define loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(mse)

# Evaluate model
correct_pred = tf.equal(tf.argmax(mse, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(tf.global_variables_initializer())
    for step in range(1, epochs+1):
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: input_data, Y: y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            R, acc = sess.run([mse, accuracy], feed_dict={X: input_data, Y: y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(logits) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: x_test,
                                      Y: y_test}))