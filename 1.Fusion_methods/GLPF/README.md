# Gaussin-Laplacian-pyramid-fusion

This code is an implementation of Gaussian and Laplacian pyramids and the fusion visible and NIR images according to [Vanmali et al article](https://www.ias.ac.in/article/fulltext/sadh/042/07/1063-1082).

## Requirements

This code works with Python 3 and requires the following libraries:
* matplotlib `pip install matplotlib`
* skimage `pip install skimage`
* numpy `pip install numpy`
* sklearn `pip install sklearn`
* scipy `pip install scipy`

## Experiments

### Classical Laplacian and Gaussian pyramids

To get the Gaussian and Laplacian pyramids of an image as well as the reconstruction of an image, run the following script:

`python main.py`

You may want to change the path of the image in the script and call the function `main_gaussian_laplacian_pyramids(image, kernel, levels)`.

Then run the following script to fill some holes in the resulted images: 
`python fill.py`

### Visible and thermal fusion using fused Laplacian pyramids

All the equations used to implement NIR and visible fusion can be found in [Vanmali et al article](https://www.ias.ac.in/article/fulltext/sadh/042/07/1063-1082).


