#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains code for making the training and test datasets.

"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tifffile


def create_dataset(folder, img_size):
    """
    This creates a tf.data.Dataset object based on all of the images that are
    within the given <folder>.  This assumes that the images are large
    (typically 1000 x 1000). This will crop a smaller image from a randomly
    chosen location, so that each iteration through will produce an image taken
    from a different part of the original file.  The label is also an image,
    the so called target.  We need to crop the same location from the target,
    to keep the image and target in sync.

    This assumes the images are tiff file format.  This assumes the target
    images have the same file name as the original image and are located in
     folder/target/samefilename.tif.

    Parameters
    ----------
    folder : str
        The name of the folder to use as the source for all of the image files.
    img_size : tuple
        The desired size (height, width) to crop the images.  The images on
        disk are large.  This function will crop the large image down to
        <img_size>.

    Returns
    -------
    dataset : tf.data.Dataset

    """
    dataset = tf.data.Dataset.list_files(folder+'/*.tif')
    dataset = dataset.map(lambda f: tuple(tfe.py_func(func=load_image,
                       inp=[f, img_size, True], Tout=(tf.float32, tf.float32))))
    return dataset


def load_image(filename, img_size, is_train):
    image = tf.py_func(tifffile.imread, [filename], [tf.uint8])
    target = tf.py_func(tifffile.imread, [filename + '.target_tif'], [tf.uint8])

    image = tf.cast(image, tf.float32)
    target = tf.cast(target, tf.float32)

    # randomly cropping to img_size[0] x img_size[1] x 3
    stacked_image = tf.stack([image, target], axis=0)
    cropped_image = tf.random_crop(stacked_image,
                                   size=[2, img_size[0], img_size[1], 3])
    image, target = cropped_image[0], cropped_image[1]

    if is_train:
        if np.random.random() > 0.5:
            # random mirror
            image = tf.image.flip_left_right(image)
            target = tf.image.flip_left_right(target)
        if np.random.random() > 0.5:
            # random flip
            image = tf.image.flip_up_down(image)
            target = tf.image.flip_up_down(target)

    # normalizing the images to [-1, 1]
    image = (image / 127.5) - 1
    target = (target / 127.5) - 1

    return image, target


if __name__ == "__main__":
    '''
    Perform a little verification testing...
    '''
    tf.enable_eager_execution()

    folder = "/shared/Projects/nuclei_segmentation/Images/Kumar_images/verification"
    ds = create_dataset(folder, img_size=(256, 256))
    out_shapes = ds.output_shapes
    ds = ds.shuffle(buffer_size=3)
    for image, target in ds:
        fig = plt.figure()
        plt.imshow(image)
        fig = plt.figure()
        plt.imshow(target)
    plt.show()
