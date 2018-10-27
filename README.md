# nuclei_segmentation
This attempts to perform nuclei segmentation of histology images using a
Conditional Adversarial Network. It uses tf.keras and eager execution to
implement the GAN following the example from the tensorflow team at
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/pix2pix/pix2pix_eager.ipynb

This creates an RGB image of where the red channel is 1.0 for all
"background" pixels; the blue channel is 1.0 for all "nuclei" pixels; and
the green channel is 1.0 for all "boundary" pixels. The boundary pixels
attempt to describe vertices of a polyline around each nuclei.


See also:
"Image-to-Image Translation with Conditional Adversarial Networks", 
https://arxiv.org/abs/1611.07004

"A dataset and a technique for generalized nuclear segmentation for computational pathology",
N. Kumar, et al, IEEE Transactions on Medical Imaging, 2017.
https://www.researchgate.net/profile/Ruchika_Verma3/publication/314271512_A_Dataset_and_a_Technique_for_Generalized_Nuclear_Segmentation_for_Computational_Pathology/links/59d5292c0f7e9b7a7e466661/A-Dataset-and-a-Technique-for-Generalized-Nuclear-Segmentation-for-Computational-Pathology.pdf
