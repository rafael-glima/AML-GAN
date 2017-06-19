
import tensorflow as tf
import numpy as np
import helper
from glob import glob

from matplotlib import pyplot
import matplotlib.image as mpimg
import os





def leaky_relu(x, alpha = 0.1):
    return tf.maximum(x*alpha, x, name= 'LeakyRelu')

def discriminator(images, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        conv1 = tf.layers.conv2d(images, 64, 5, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv1 = tf.layers.batch_normalization(conv1, training=True)
        conv1 = leaky_relu(conv1)

        conv2 = tf.layers.conv2d(conv1, 128, 5, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv2 = tf.layers.batch_normalization(conv2, training=True)
        conv2 = leaky_relu(conv2)

        conv3 = tf.layers.conv2d(conv2, 256, 5, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv3 =tf.layers.batch_normalization(conv3, training=True)
        conv3 = leaky_relu(conv3)

        conv4 = tf.layers.conv2d(conv3, 512, 5, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv4 = tf.layers.batch_normalization(conv4, training=True)
        conv4 = leaky_relu(conv4)

        flat = tf.contrib.layers.flatten(conv4)

        logits = tf.layers.dense(flat, 1)

        out = tf.sigmoid(logits)

        return out, logits

def generator(z, out_chanel_dim, is_train=True):

    with tf.variable_scope('generator', reuse= not  is_train):

        conv1 = tf.layers.dense(z, 4*4*1024)

        conv1 = tf.reshape(conv1, (-1,4,4,1024))
        conv1 = tf.layers.batch_normalization(conv1, training=is_train)
        conv1 = leaky_relu(conv1)

        conv2 = tf.layers.conv2d_transpose(conv1, 512, 5, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv2 = tf.layers.batch_normalization(conv2, training=is_train)
        conv2 = leaky_relu(conv2)

        conv3 = tf.layers.conv2d_transpose(conv2, 256, 5, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv3 = tf.layers.batch_normalization(conv3, training=is_train)
        conv3 = leaky_relu(conv3)

        conv4 = tf.layers.conv2d_transpose(conv3, 128, 5, strides=2, padding='same', kernel_initializer= tf.contrib.layers.xavier_initializer())
        conv4 = tf.layers.batch_normalization(conv4, training=is_train)
        conv4 = leaky_relu(conv4)

        logits = tf.layers.conv2d_transpose(conv4, out_chanel_dim, 5, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer())

        logits = tf.image.resize_images(logits, (28,28))

        out = tf.tanh(logits)

        return out

def model_loss(input_real, input_z, out_chanel_dim, smooth = 0.1):

    g_model = generator(input_z, out_chanel_dim)
    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                labels=tf.ones_like(d_model_real)*(1 - smooth)))

    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.ones_like(d_model_fake)
                                                )
    )

    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.ones_like(d_model_fake)
                                                )
    )

    return d_loss, g_loss

def model_opt(d_loss, g_loss, learning_rate, beta1):

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt



def show_generator_output(sess, n_images, input_z, output_chanel_dim, image_mode, path,plot=False):

    cmap = None if image_mode== 'RGB' else 'gray'

    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1,1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, output_chanel_dim, False),
        feed_dict = {input_z: example_z}
    )

    images_grid = helper.images_square_grid(samples, image_mode)

    if plot:
        pyplot.show(images_grid, cmap=cmap)
        pyplot.show()

    mpimg.imsave(path, images_grid)


def model_inputs(image_witdh, image_height, image_chanels, z_dim):

    input_real = tf.placeholder(tf.float32, (None, image_witdh, image_height, image_chanels), name='input_real')
    input_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    learning_rate = tf.placeholder(tf.float32, (None), name='learning_rate')

    return input_real, input_z, learning_rate


def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode, path):

    input_real, input_z, lr = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)

    d_loss, g_loss = model_loss(input_real, input_z, data_shape[3])
    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate, beta1)

    step = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):

            for batch_images in get_batches(batch_size):

                step += 1

                batch_z = np.random.uniform(-1,1, size=(batch_size, z_dim))
                batch_images = batch_images *2

                _ = sess.run(d_opt, feed_dict = {input_real: batch_images, input_z:batch_z, lr: learning_rate})
                _ = sess.run(g_opt, feed_dict = {input_z: batch_z, input_real:batch_images})

                if step % 10 == 0:
                    train_loss_d = d_loss.eval({input_z: batch_z, input_real:batch_images})
                    train_loss_g = g_loss.eval({input_z: batch_z})
                    print("- epoch {}/{} ".format(epoch_i+1, epoch_count),
                          "| discriminator loss: {:.4f}...".format(train_loss_d),
                          "| generator loss: {:.4f}".format(train_loss_g))



data_dir = '../data'
helper.download_extract('celeba', data_dir)


batch_size = 64
z_dim = 100
learning_rate = 0.0002
beta1 = 0.5

path = 'generated_celeba'
#
os.makedirs(path)

epochs = 100

mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
          mnist_dataset.shape, mnist_dataset.image_mode, path)

