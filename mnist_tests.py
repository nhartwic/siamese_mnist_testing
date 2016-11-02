from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 2]))
b = tf.Variable(tf.zeros([2]))

sess.run(tf.initialize_all_variables())

y_points = tf.matmul(x,W) + b
print(y_points.get_shape())
# y_diff = tf.slice(y_points, y_points.get_shape())
