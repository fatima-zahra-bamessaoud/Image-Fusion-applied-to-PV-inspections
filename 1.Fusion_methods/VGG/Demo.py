import numpy as np
from imageio import imread
import glob
import os
import torch
from skimage.color import rgb2gray
from torchvision.models.vgg import vgg19
import cv2

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
    gray = imread(path1)

    # VI image
    VjpgFileName = os.path.basename(VjpgFiles[k])
    path2 = os.path.join(VFolderPath, VjpgFileName)
    ir = imread(path2)

    # Assuming you have a 'fuse' function defined somewhere
    from vggfusion import fuse

    fused_image = fuse(gray, ir)

    from vggfusion import *
    npad = 16
    lda = 5
    graylow, grayhigh = lowpass(gray.astype(np.float32) / 255, lda, npad)
    irlow, irhigh = lowpass(ir.astype(np.float32) / 255, lda, npad)
    grayhigh3 = c3(grayhigh)
    irhigh3 = c3(irhigh)

    model = vgg19(True).cpu().eval()
    gray_in = torch.from_numpy(grayhigh3).cpu()
    ir_in = torch.from_numpy(irhigh3).cpu()
    relus = [2, 7, 12, 21]
    unit_relus = [1, 2, 4, 8]
    relus_gray = get_activation(model, relus, gray_in)
    relus_ir = get_activation(model, relus, ir_in)
    gray_feats = [l1_features(out) for out in relus_gray]
    ir_feats = [l1_features(out) for out in relus_ir]
    saliencies = []
    saliency_max = None

    for idx in range(len(relus)):
        saliency_current = fusion_strategy(
            gray_feats[idx], ir_feats[idx], grayhigh, irhigh, unit_relus[idx])
        saliencies.append(saliency_current)
        if saliency_max is None:
            saliency_max = saliency_current
        else:
            saliency_max = np.maximum(saliency_max, saliency_current)

    low_fused = (graylow + irlow) / 2
    high_fused = saliency_max
    fusion = low_fused + high_fused

    # Save the fused image
    output_path = f'output/fused-images/VGG-fused_{k+5}.JPG'
    gray_fused = cv2.cvtColor((fusion * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path, gray_fused)

    print(f"Processed and saved: VGG-fused_{k}.JPG")

print("Processing complete.")
