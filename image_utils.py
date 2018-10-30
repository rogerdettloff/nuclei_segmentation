#!/usr/bin/env python
"""
This module contains tools that have been helpful for working with histological
image files, including reading the image annotation files, making the target
image.

"""

import cv2
import xml.etree.ElementTree as et
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import tifffile


def read_annos_xml(filename):
    """
    This reads the annotations for one image of the Kumar et al. (2018) dataset
    and returns the x and y vertices (pixel locations) that trace the region
    around each cell nucleus. The return values are each an ndarray where each
    element is an ndarray of vertices outlining one nucleus region.

    :param str filename:
    :return ndarray x_vert, ndarray y_vert:
    """
    tree = et.parse(filename)
    root = tree.getroot()

    x_vert = []
    y_vert = []
    for regions in root.iter('Regions'):
        for region in regions.iter('Region'):
            x_region = []
            y_region = []
            for vertex in region.iter('Vertex'):
                x_region.append(float(vertex.attrib['X']))
                y_region.append(float(vertex.attrib['Y']))
            x_vert.append(np.round(np.asarray(x_region, dtype=int)))
            y_vert.append(np.round(np.asarray(y_region, dtype=int)))

    return np.asarray(x_vert), np.asarray(y_vert)


def _make_target_image(x_vert, y_vert, im_size):
    """
    This create an RGB image of im_size where the red channel is 1.0 for all
    "background" pixels; the blue channel is 1.0 for all "nuclei" pixels; and
    the green channel is 1.0 for all "boundary" pixels. The boundary pixels
    are described by x_vert, y_vert vertices of a polyline around each nuclei.

    This so called target image defines the classification for each pixel in
    the original image.  The target image is the desired result that will be
    used to train a decoder to classify all the pixels of the original image.

    :param ndarray x_vert: x-component of vertices of polyline around each
        nuclei boundary.  Obtained from read_annos_xml().
    :param ndarray y_vert:  y-component of vertices of polyline around each
        nuclei boundary.  Obtained from read_annos_xml().
    :param tuple im_size:
    :return: ndarray im
    """

    im = np.zeros(im_size + (3,))   # RGB image canvas

    max_x = im_size[0] - 1
    max_y = im_size[1] - 1
    for n in range(len(x_vert)):
        x = x_vert[n]
        y = y_vert[n]
        # truncate any nucli boundaries that go past the edge of the image.
        x[x < 0] = 0
        x[x > max_x] = max_x
        y[y < 0] = 0
        y[y > max_y] = max_y

        # prepare the vertex points for input to openCV polylines()
        pts = np.stack([x, y], axis=1).reshape((-1, 1, 2))

        # draw the "boundary" outlines in the green channel
        cv2.polylines(im, [pts], isClosed=True,
                      color=(0.0, 1.0, 0.0),
                      thickness=5,
                      lineType=cv2.LINE_8
                      )
        # draw the initial "nuclei" region in the blue channel
        cv2.fillPoly(im, [pts],
                      color=(0.0, 0.0, 1.0),  # RGB
                      lineType=cv2.LINE_8
                      )

    # make the "background" in red channel by subtracting the "nuclei"
    # region from ones.
    im[:, :, 0] = np.ones(im_size) - im[:, :, 2]

    # finally subtract the "boundary" from the "nuclei" to produce the
    # "inside" region in the blue channel, and subtract the "boudary" from the
    # "background" in red.
    im[:, :, 2] = np.maximum(im[:, :, 2] - im[:, :, 1], 0.0)
    im[:, :, 0] = np.maximum(im[:, :, 0] - im[:, :, 1], 0.0)

    return im


def make_target_images(anno_folder, target_folder, im_size):
    """
    This will make a target image for each of the annotation files found in
    <anno_folder>.  Resulting target images are written as tiff files to
    <target_folder>.

    :param str anno_folder:
    :param str target_folder:
    :param tuple im_size:
    :return: none
    """
    filenames = glob(os.path.join(anno_folder, '*.xml'), recursive=False)
    for anno_file in filenames:
        x_vert, y_vert = read_annos_xml(anno_file)
        im_target = _make_target_image(x_vert, y_vert, im_size)
        # convert to uint8...
        im_target = (im_target*255).astype(dtype=np.uint8)
        filename, _ = os.path.splitext(os.path.basename(anno_file))
        target_filename = os.path.join(target_folder, filename+'.tif')
        tifffile.imsave(target_filename, im_target, photometric='rgb')



if __name__ == "__main__":
    '''
    This performs a small verification test...
    '''
    path = '/shared/Projects/nuclei_segmentation/Images/Kumar_images/'
    anno_file = os.path.join(path, 'Annotations/TCGA-18-5592-01Z-00-DX1.xml')
    image_file = os.path.join(path, 'deflated/TCGA-18-5592-01Z-00-DX1.tif')

    anno_folder = os.path.join(path, 'Annotations/')
    target_folder = os.path.join(path, 'target/')
    make_target_images(anno_folder, target_folder, (1000,1000))

    x_vert, y_vert = read_annos_xml(anno_file)
    im_slide = tifffile.imread(image_file)
    im_size = im_slide.shape[0:2]
    im_target = _make_target_image(x_vert, y_vert, im_size)
    # copy just the boundary image and make everything else transparent...
    im_boundary = np.zeros(im_size + (4,))  # RGBA canvas
    im_boundary[:, :, 1] = im_target[:, :, 1]  # copy the green boundary channel
    im_boundary[:, :, 3] = im_target[:, :, 1]  # alpha channel...boundary opaque

    plt.imshow(im_slide)
    plt.imshow(im_boundary)  # overlay the boundary on the slide for verification
    plt.show()

    plt.imshow(im_target)  # plot target image in a new figure.
    plt.show()
    pass

