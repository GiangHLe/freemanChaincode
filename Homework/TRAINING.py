from __future__ import print_function
import numpy as np
import tensorflow as tf
class XOR(object):
    def __init__(**kwags):

# Parameters
learning_rate = 1
num_steps = 500
batch_size = 128
display_step = 100

n_hidden = 2 # 1st layer number of neurons
num_input = 2 
num_classes = 2 

input_data = np.array([[1,1],[-1,-1],[1,-1],[-1,1]])
# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h': tf.Variable(tf.random_normal([num_input, zn_hidden_1],mean = 0, stddev = 0.02, dtype = tf.dtypes.float32, seed = 200)),
    'w_out': tf.Variable(tf.random_normal([n_hidden_2, num_classes], mean = 0, stddev = 0.02, dtype = tf.dtypes.float32, seed = 200))
}
biases = {
    'b': tf.zeros([n_hidden_1], dtype = tf.dtypes.float32),
    'b_out': tf.zeros([num_classes], dtype = tf.dtypes.float32)
}


# Create model
def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

logits = neural_net(X)

prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))