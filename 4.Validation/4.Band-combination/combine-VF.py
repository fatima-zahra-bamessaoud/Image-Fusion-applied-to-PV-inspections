import cv2
import numpy as np

for i in range(1, 500):
    # Load the fused grayscale image
    fused_image = cv2.imread(rf'C:\Users\DELL-DK-STOR\Desktop\Working\4.Validation\1.Fusion\MGFF\output\MGFF-fused_{i}.JPG', cv2.IMREAD_GRAYSCALE)

    # Load the visible RGB image
    visible_image = cv2.imread(rf'C:\Users\DELL-DK-STOR\Desktop\Working\4.Validation\1.Fusion\1.source\V-images\DJI_{i}_V.JPG', cv2.IMREAD_COLOR)
                                
    # Stack the arrays to create a four-band image
    FRGB_image = np.dstack((fused_image, visible_image))
    RGBF_image = np.dstack((visible_image ,fused_image))

    # Save the multi-band image
    cv2.imwrite(rf'C:\Users\DELL-DK-STOR\Desktop\Working\4.Validation\4.Band-combination\FRGB\FRGB-{i}.TIFF', FRGB_image)
    cv2.imwrite(rf'C:\Users\DELL-DK-STOR\Desktop\Working\4.Validation\4.Band-combination\RGBF\RGBF-{i}.TIFF', RGBF_image)

