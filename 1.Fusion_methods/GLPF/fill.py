import cv2
import numpy as np

for k in range(5,16):
    # Load the fused (grayscale) image and the infrared (grayscale) image
    fused_image = cv2.imread(rf'C:\Users\DELL-DK-STOR\Desktop\Working\1.Fusion_methods\GLPF\output\fused-images-brute\GLPF_{k}.JPG', cv2.IMREAD_GRAYSCALE)
    infrared_image = cv2.imread(rf'C:\Users\DELL-DK-STOR\Desktop\Working\1.Fusion_methods\1.source\T-alignedF\DJI_{k}_T.JPG', cv2.IMREAD_GRAYSCALE)

    # Ensure both images have the same spatial dimensions
    if fused_image.shape != infrared_image.shape:
        raise ValueError("The fused image and infrared image must have the same width and height.")

    # Navigate over each pixel of the fused image
    rows, cols = fused_image.shape
    for i in range(rows):
        for j in range(cols):
            # Detect holes in the fused image (pixel value 50)
            if fused_image[i, j] < 50:
                # Replace the hole pixel with the corresponding pixel value from the infrared image
                fused_image[i, j] = infrared_image[i, j]
                
                

    # Save the result
    cv2.imwrite(rf'C:\Users\DELL-DK-STOR\Desktop\Working\1.Fusion_methods\GLPF\output\fused-images\GLPF-fused_{k}.JPG', fused_image)


