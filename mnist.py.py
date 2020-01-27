import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/',one_hot=True)

nodes_hl1 = 500
nodes_hl2 = 500
nodes_hl3 = 500
nodes_hl4 = 500

num_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):

    initializer_xavier = tf.contrib.layers.xavier_initializer()

    hidden_layer1 = {'weights': tf.Variable(initializer_xavier([784, nodes_hl1])),
                     'biases': tf.Variable(initializer_xavier([nodes_hl2]))}

    hidden_layer2 = {'weights': tf.Variable(initializer_xavier([nodes_hl1, nodes_hl2])),
                     'biases': tf.Variable(initializer_xavier([nodes_hl2]))}

    hidden_layer3 = {'weights': tf.Variable(initializer_xavier([nodes_hl2, nodes_hl3])),
                     'biases': tf.Variable(initializer_xavier([nodes_hl3]))}

    hidden_layer4 = {'weights': tf.Variable(initializer_xavier([nodes_hl3, nodes_hl4])),
                     'biases': tf.Variable(initializer_xavier([nodes_hl4]))}

    output_layer = {'weights': tf.Variable(initializer_xavier([nodes_hl4, num_classes])),
                    'biases': tf.Variable(initializer_xavier([num_classes]))}

    hl1 = tf.add(tf.matmul(data, hidden_layer1['weights']), hidden_layer1['biases'])
    hl1 = tf.nn.relu(hl1)

    hl2 = tf.add(tf.matmul(hl1, hidden_layer2['weights']), hidden_layer2['biases'])
    hl2 = tf.nn.relu(hl2)

    hl3 = tf.add(tf.matmul(hl2, hidden_layer3['weights']), hidden_layer3['biases'])
    hl3 = tf.nn.relu(hl3)

    hl4 = tf.add(tf.matmul(hl3, hidden_layer4['weights']), hidden_layer4['biases'])
    hl4 = tf.nn.relu(hl4)

    output = tf.add(tf.matmul(hl4, output_layer['weights']), output_layer['biases'])

    return output

def train_neural_network(x):
    predictions = neural_network_model(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    num_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print("Epoch", epoch+1,"completed of", num_epochs, "Loss:", epoch_loss)

        correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print("Accuracy of the model: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)