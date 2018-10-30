#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This attempts to perform nuclei segmentation of histology images using a
Conditional Adversarial Network. It uses tf.keras and eager execution to
implement the GAN following the example from the tensorflow team at
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/pix2pix/pix2pix_eager.ipynb

See also:
"Image-to-Image Translation with Conditional Adversarial Networks",
    https://arxiv.org/abs/1611.07004

"""

from inputs import create_dataset
import matplotlib.pyplot as plt
import tensorflow as tf
import time


class Downsample(tf.keras.Model):
    """
    This creates an encoding layer consisting of conv2d + batchNormalization
    with leaky_relu activation.
    """
    def __init__(self, filters, size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Upsample(tf.keras.Model):
    """
    This creates a decoder layer using deconvolution + batchNormalization +
    optional dropout with a skip connection from back in the encoder.
    """
    def __init__(self, filters, size, apply_dropout=False):
        super(Upsample, self).__init__()
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)

        self.up_conv = tf.keras.layers.Conv2DTranspose(filters,
                                                       (size, size),
                                                       strides=2,
                                                       padding='same',
                                                       kernel_initializer=initializer,
                                                       use_bias=False)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x1, x2, training):
        x = self.up_conv(x1)
        x = self.batchnorm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        x = tf.nn.relu(x)
        # bring in the skip connection
        x = tf.concat([x, x2], axis=-1)
        return x


class Generator(tf.keras.Model):
    """
    Create the generator network.  The architecture of generator is a modified
    U-Net. Each block in the encoder is (Conv -> Batchnorm -> Leaky ReLU).
    Each block in the decoder is (Transposed Conv -> Batchnorm ->
    Dropout(applied to the first 3 blocks) -> ReLU).  There are skip
    connections between the encoder and decoder (as in U-Net).
    """
    def __init__(self, num_output_channels):
        super(Generator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
            # Let's say the input is for example (256 x 256)
        self.down1 = Downsample(64, 4, apply_batchnorm=False)  # (128 x 128)
        self.down2 = Downsample(128, 4)     # (64 x 64)
        self.down3 = Downsample(256, 4)     # (32 x 32)
        self.down4 = Downsample(512, 4)     # (16 x 16)
        self.down5 = Downsample(512, 4)     # (8 x 8)
        self.down6 = Downsample(512, 4)     # (4 x 4)
        self.down7 = Downsample(512, 4)     # (2 x 2)
        self.down8 = Downsample(512, 4)     # (1 x 1)

        self.up1 = Upsample(512, 4, apply_dropout=True)     # (2 x 2)
        self.up2 = Upsample(512, 4, apply_dropout=True)     # (4 x 4)
        self.up3 = Upsample(512, 4, apply_dropout=True)     # (8 x 8)
        self.up4 = Upsample(512, 4)     # (16 x 16)
        self.up5 = Upsample(256, 4)     # (32 x 32)
        self.up6 = Upsample(128, 4)     # (64 x 64)
        self.up7 = Upsample(64, 4)      # (128 x 128)

        self.last = tf.keras.layers.Conv2DTranspose(    # (256 x 256)
            num_output_channels,
            (4, 4),
            strides=2,
            padding='same',
            kernel_initializer=initializer
        )

    @tf.contrib.eager.defun   # Remember, functions compiled with defun cannot be inspected with pdb
    def call(self, x, training):
        # x shape == (bs, 256, 256, 3)
        x1 = self.down1(x, training=training) # (bs, 128, 128, 64)
        x2 = self.down2(x1, training=training) # (bs, 64, 64, 128)
        x3 = self.down3(x2, training=training) # (bs, 32, 32, 256)
        x4 = self.down4(x3, training=training) # (bs, 16, 16, 512)
        x5 = self.down5(x4, training=training) # (bs, 8, 8, 512)
        x6 = self.down6(x5, training=training) # (bs, 4, 4, 512)
        x7 = self.down7(x6, training=training) # (bs, 2, 2, 512)
        x8 = self.down8(x7, training=training) # (bs, 1, 1, 512)

        x9 = self.up1(x8, x7, training=training) # (bs, 2, 2, 1024)
        x10 = self.up2(x9, x6, training=training) # (bs, 4, 4, 1024)
        x11 = self.up3(x10, x5, training=training) # (bs, 8, 8, 1024)
        x12 = self.up4(x11, x4, training=training) # (bs, 16, 16, 1024)
        x13 = self.up5(x12, x3, training=training) # (bs, 32, 32, 512)
        x14 = self.up6(x13, x2, training=training) # (bs, 64, 64, 256)
        x15 = self.up7(x14, x1, training=training) # (bs, 128, 128, 128)

        x16 = self.last(x15) # (bs, 256, 256, 3)
        x16 = tf.nn.tanh(x16)

        return x16


class DiscDownsample(tf.keras.Model):

    def __init__(self, filters, size, apply_batchnorm=True):
        super(DiscDownsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)

        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x


class Discriminator(tf.keras.Model):
    """
    Create the Discriminator network.  The Discriminator is a PatchGAN.
    Each block in the discriminator is (Conv -> BatchNorm -> Leaky ReLU).
    The shape of the output after the last layer is (batch_size, 30, 30, 1)
    Each 30x30 patch of the output classifies a 70x70 portion of the input
    image (such an architecture is called a PatchGAN).  The Discriminator
    receives 2 inputs, the Input image and the target image, which it should
    classify as real.  The Input image and the generated image (output of
    generator), which it should classify as fake.  We concatenate these 2
    inputs together in the code (tf.concat([inp, tar], axis=-1)).  The shape of
    the input travelling through the generator and the discriminator is in the
    comments in the code.

    """
    def __init__(self):
        super(Discriminator, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)

        self.down1 = DiscDownsample(64, 4, False)
        self.down2 = DiscDownsample(128, 4)
        self.down3 = DiscDownsample(256, 4)

        # we are zero padding here with 1 because we need our shape to
        # go from (batch_size, 32, 32, 256) to (batch_size, 31, 31, 512)
        self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
        self.conv = tf.keras.layers.Conv2D(512,
                                           (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer,
                                           use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        # shape change from (batch_size, 31, 31, 512) to (batch_size, 30, 30, 1)
        self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
        self.last = tf.keras.layers.Conv2D(1,
                                           (4, 4),
                                           strides=1,
                                           kernel_initializer=initializer)

    @tf.contrib.eager.defun     # Remember, functions compiled with defun cannot be inspected with pdb
    def call(self, inp, tar, training):
        # concatenating the input and the target
        x = tf.concat([inp, tar], axis=-1)  # (bs, 256, 256, channels*2)
        x = self.down1(x, training=training)  # (bs, 128, 128, 64)
        x = self.down2(x, training=training)  # (bs, 64, 64, 128)
        x = self.down3(x, training=training)  # (bs, 32, 32, 256)

        x = self.zero_pad1(x)  # (bs, 34, 34, 256)
        x = self.conv(x)  # (bs, 31, 31, 512)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.zero_pad2(x)  # (bs, 33, 33, 512)
        # don't add a sigmoid activation here since
        # the loss function expects raw logits.
        x = self.last(x)  # (bs, 30, 30, 1)

        return x


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(disc_real_output),
        logits=disc_real_output
    )
    generated_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(disc_generated_output),
        logits=disc_generated_output
    )

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generator_loss(disc_generated_output, gen_output, target):
    LAMBDA = 100
    gan_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(disc_generated_output),
        logits=disc_generated_output
    )
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss


def train(dataset, test_dataset, epochs, checkpoint_prefix):
    generator = Generator(num_output_channels=3)
    discriminator = Discriminator()

    generator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)
    discriminator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )

    with tf.contrib.summary.create_file_writer(
            '../logs').as_default(), tf.contrib.summary.always_record_summaries():
        step = 0
        for epoch in range(epochs):
            start = time.time()

            for input_image, target in dataset:
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    gen_output = generator(input_image, training=True)

                    disc_real_output = discriminator(input_image, target,
                                                     training=True)
                    disc_generated_output = discriminator(input_image, gen_output,
                                                          training=True)

                    gen_loss = generator_loss(disc_generated_output, gen_output,
                                              target)
                    disc_loss = discriminator_loss(disc_real_output,
                                                   disc_generated_output)

                tf.contrib.summary.scalar("generator_loss", gen_loss, step=step)
                tf.contrib.summary.scalar("discriminator_loss", disc_loss, step=step)
                step = step+1

                generator_gradients = gen_tape.gradient(gen_loss,
                                                        generator.variables)
                discriminator_gradients = disc_tape.gradient(disc_loss,
                                                             discriminator.variables)

                generator_optimizer.apply_gradients(zip(generator_gradients,
                                                        generator.variables))
                discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                            discriminator.variables))

            if epoch % 240 == 0:
                for inp, tar in test_dataset.take(1):
                    test_predictions(generator, inp, tar)

            # saving (checkpoint) the model every so often
            if (epoch + 1) % 100 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                               time.time() - start))
            print('gen_loss: {};  disc_loss: {} \n'.format(gen_loss, disc_loss))

        tf.contrib.summary.flush()


def test_predictions(model, test_input, tar):
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Expert Segmentation', 'Predicted Segmentation']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    '''
    Perform a little verification testing...
    '''
    tf.enable_eager_execution()

    img_size = (256, 256)
    test_folder = "/shared/Projects/nuclei_segmentation/Images/Kumar_images/verification"
    test_ds = create_dataset(test_folder, img_size=img_size)
    test_ds = test_ds.batch(1)  # this creates a 4-dim tensor, like: (1, 256, 256, 3)
    train_folder = "/shared/Projects/nuclei_segmentation/Images/Kumar_images/training"
    train_ds = create_dataset(test_folder, img_size=img_size)
    train_ds = train_ds.shuffle(buffer_size=5)
    train_ds = train_ds.batch(1)

    checkpoint_dir = "/shared/Projects/nuclei_segmentation/models/"
    train(train_ds, test_ds, epochs=2400, checkpoint_prefix=checkpoint_dir)

