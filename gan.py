
from tensorflow.contrib import layers, losses
from tensorflow.contrib.framework import arg_scope
import tensorflow as tf
from scipy.misc import imsave



def concat_elu(inputs):
    return tf.nn.elu(tf.concat([-inputs, inputs], 3))

def encoder(input_tensor, output_size):

    net = tf.reshape(input_tensor, [-1, 28, 28, 1])
    net = layers.conv2d(net, 32, 5, stride=2)
    net = layers.conv2d(net, 64, 5, stride=2)
    net = layers.conv2d(net, 128, 5, stride=2, padding='VALID')
    net = layers.dropout(net, keep_prob=0.9)
    net = layers.flatten(net)

    return layers.fully_connected(net, output_size, activation_fn=None)

def discriminator(input_tensor):
    return encoder(input_tensor, 1)

def decoder(input_tensor):

    net = tf.expand_dims(input_tensor, 1)
    net = tf.expand_dims(net, 1)
    net = layers.conv2d_transpose(net, 128, 3, padding='VALID')
    net = layers.conv2d_transpose(net, 64, 5, padding='VALID')
    net = layers.conv2d_transpose(net, 32, 5, stride=2)
    net = layers.conv2d_transpose(
        net, 1, 5, stride=2, activation_fn=tf.nn.sigmoid
    )
    net = layers.flatten(net)
    return net

class GenerativeAdversarialNet(object):

    def __init__(self, hidden_size, batch_size, learning_rate):

        self.input_tensor = tf.placeholder(tf.float32, [None, 28*28])

        with arg_scope([layers.conv2d, layers.conv2d_transpose],
            activation_fn=concat_elu,
            normalizer_fn = layers.batch_norm,
            normalizer_params={'scale':True}):

            with tf.variable_scope("model"):
                D1 = discriminator(self.input_tensor)
                D_param_num = len(tf.trainable_variables())
                G = decoder(tf.random_normal([batch_size, hidden_size]))

                self.sampled_tensor = G

            with tf.variable_scope("model", reuse=True):
                D2 = discriminator(G)

        discriminator_loss = self.__get_discriminator_loss(D1,D2)

        G_loss = self.__get_generator_loss(D2)

        params = tf.trainable_variables()

        D_params = params[:D_param_num]
        G_params = params[D_param_num:]

        global_step = tf.contrib.framework.get_or_create_global_step()

        self.train_discriminator = layers.optimize_loss(discriminator_loss, global_step, learning_rate/10, 'Adam', variables=D_params, update_ops=[])
        self.train_generator = layers.optimize_loss(G_loss, global_step, learning_rate, 'Adam', variables=G_params, update_ops=[])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def __get_discriminator_loss(self, D1, D2):

        return (losses.sigmoid_cross_entropy(D1, tf.ones(tf.shape(D1)))  +
                losses.sigmoid_cross_entropy(D2, tf.zeros(tf.shape(D1)))
                )

    def __get_generator_loss(self, D2):

        return losses.sigmoid_cross_entropy(D2, tf.ones(tf.shape(D2)))

    def update_params(self, inputs):

        d_loss_value = self.sess.run(self.train_discriminator, {
            self.input_tensor: inputs
        })

        g_loss_value = self.sess.run(self.train_generator)

        return g_loss_value

    def generate_and_save_images(self, num_samples, directory):

        imgs = self.sess.run(self.sampled_tensor)
        for k in range(imgs.shape[0]):
            imgs_folder = os.path.join(directory, 'imgs')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder)

            imsave(os.path.join(imgs_folder, '%d.png') % k,imgs[k].reshape(28, 28))

flags = tf.flags
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "the number of udpates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("model", "gan", "gan only")

FLAGS = flags.FLAGS

if __name__ == "__main__":

    import os
    from tensorflow.examples.tutorials.mnist import input_data
    from progressbar import ETA, Bar, Percentage, ProgressBar

    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    mnist = input_data.read_data_sets(data_directory, one_hot=True)

    model = GenerativeAdversarialNet(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)

    for epoch in range(FLAGS.max_epoch):

        training_loss = 0.0

        pbar = ProgressBar()
        for i in pbar(range(FLAGS.updates_per_epoch)):
            images, _ = mnist.train.next_batch(FLAGS.batch_size)
            loss_value = model.update_params(images)
            training_loss += loss_value

        training_loss = training_loss / (FLAGS.updates_per_epoch * FLAGS.batch_size)

        print("Loss %f" % training_loss)

        model.generate_and_save_images(FLAGS.batch_size, FLAGS.working_directory)








