# nuclei_segmentation
This attempts to perform nuclei segmentation of histology images using a Conditional Adversarial Network. It uses tf.keras and eager execution to implement the patch-GAN following the example from the tensorflow team: [pix2pix_eager.ipynb](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/pix2pix/pix2pix_eager.ipynb)

This creates an RGB image where the red channel is 1.0 for all "background" pixels; the blue channel is 1.0 for all "nuclei" pixels; and the green channel is 1.0 for all "boundary" pixels. The boundary pixels attempt to describe vertices of a polyline around each nuclei.

See also:
[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) 

[Kumar, et al, A dataset and a technique for generalized nuclear segmentation for computational pathology](https://www.researchgate.net/profile/Ruchika_Verma3/publication/314271512_A_Dataset_and_a_Technique_for_Generalized_Nuclear_Segmentation_for_Computational_Pathology/links/59d5292c0f7e9b7a7e466661/A-Dataset-and-a-Technique-for-Generalized-Nuclear-Segmentation-for-Computational-Pathology.pdf)

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

### Example images illustrating promising results:
![image](https://user-images.githubusercontent.com/6138503/47751436-fc1b2f80-dc4e-11e8-853f-a2412a28bd2e.png)

![image](https://user-images.githubusercontent.com/6138503/48308884-3bd0f980-e522-11e8-9c62-fe0483cb5cf8.png)
