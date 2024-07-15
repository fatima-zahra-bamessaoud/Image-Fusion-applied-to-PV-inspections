import matplotlib.pyplot as plt
import glob
import os 
from pyramids import *
from weight_map import *
import cv2
import subprocess
from skimage.color import rgb2hsv, rgb2gray
from skimage.transform import resize

def main_multimodal_fusion(im_vis, im_ir, kernel, levels, window_size):
    """
    A function to fuse two images of different modalities, in this example we use visible and NIR images.

    :param im_vis: The visible image, a numpy array of floats within [0, 1] of shape (N, M, 3)
    :param im_ir: The NIR image, a numpy array of floats within [0, 1] of shape (N, M)
    :param kernel: The Gaussian kernel used to build pyramids
    :param levels: The desired levels in the pyramids
    :param window_size: The window size used to compute the local entropy and the local contrast
    """

    im_vis = convert_image_to_floats(im_vis)
    im_ir = convert_image_to_floats(im_ir)

    im_vis_hsv = rgb2hsv(im_vis)
    value_channel = im_vis_hsv[:, :, 2]
    im_ir_gray = rgb2gray(im_ir)

    # kernels to compute visibility
    kernel1 = classical_gaussian_kernel(5, 2)
    kernel2 = classical_gaussian_kernel(5, 2)

    # Computation of local entropy, local contrast and visibility for value channel
    local_entropy_value = normalized_local_entropy(value_channel, window_size)
    local_contrast_value = local_contrast(value_channel, window_size)
    visibility_value = visibility(value_channel, kernel1, kernel2)
    # Combination of local entropy, local contrast and visibility for value channel
    weight_value = weight_combination(local_entropy_value, local_contrast_value, visibility_value, 1, 1, 1)

    # Computation of local entropy, local contrast and visibility for IR image
    local_entropy_ir = normalized_local_entropy(im_ir_gray, window_size)
    local_contrast_ir = local_contrast(im_ir_gray, window_size)
    visibility_ir = visibility(im_ir_gray, kernel1, kernel2)
    # Combination of local entropy, local contrast and visibility for IR image
    weight_ir = weight_combination(local_entropy_ir, local_contrast_ir, visibility_ir, 1, 1, 1)
     
    # Normalising weights of value channel and IR image
    weightN_value, weightN_ir = weight_normalization(weight_value, weight_ir)

    # Creating Gaussian pyramids of the weights maps of respectively the value channel and IR image
    gauss_pyr_value_weights = gaussian_pyramid(weightN_value, kernel, levels)
    gauss_pyr_ir_weights = gaussian_pyramid(weightN_ir, kernel, levels)

    # Creating Laplacian pyramids of respectively the value channel and IR image
    lap_pyr_value = laplacian_pyramid(value_channel, kernel, levels)
    lap_pyr_ir = laplacian_pyramid(im_ir_gray, kernel, levels)

    # Creating the fused Laplacian of the two modalities
    lap_pyr_fusion = fused_laplacian_pyramid(gauss_pyr_value_weights, gauss_pyr_ir_weights, lap_pyr_value, lap_pyr_ir)

    # Creating the Gaussian pyramid of value channel in order to collapse the fused Laplacian pyramid
    gauss_pyr_value = gaussian_pyramid(value_channel, kernel, levels)
    collapsed_image = collapse_pyramid(lap_pyr_fusion, gauss_pyr_value)
    
    return collapsed_image


def convert_image_to_floats(image):
    """
    A function to convert an image to a numpy array of floats within [0, 1]

    :param image: The image to be converted
    :return: The converted image
    """

    if np.max(image) <= 1.0:
        return image
    else:
        return image / 255.0


kernel = smooth_gaussian_kernel(0.4)
levels = 4
window_size = 5


# Define the folder path of thermal images
TFolderPath = 'C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/1.source/T-alignedF'
# List all JPG files in the current folder
TjpgFiles = glob.glob(os.path.join(TFolderPath, '*.JPG'))

# Define the folder path of visible images
VFolderPath = 'C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/1.source/V-alignedF'
# List all JPG files in the current folder
VjpgFiles = glob.glob(os.path.join(VFolderPath, '*.JPG'))


for k in range(len(TjpgFiles)):
    # IR image
    TjpgFileName = os.path.basename(TjpgFiles[k])
    path1 = os.path.join(TFolderPath, TjpgFileName)

    # VI image
    VjpgFileName = os.path.basename(VjpgFiles[k])
    path2 = os.path.join(VFolderPath, VjpgFileName)
    
    
    image_full = plt.imread(path1)     #thermal
    image2_full = plt.imread(path2)      #RGB

    image = resize(image_full, (int(image_full.shape[0]), int(image_full.shape[1])))
    image2_full_rs = resize(image2_full, (int(image2_full.shape[0]), int(image2_full.shape[1]), 3))

    fusion = main_multimodal_fusion(image2_full_rs, image, kernel, levels, window_size)

    output_path = f'C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/GLPF/output/fused-images-brute/GLPF_{k+5}.JPG'
    cv2.imwrite(output_path, (fusion * 255).astype(np.uint8))
    
    
  