import cv2
import numpy as np

for i in range(1, 500):
    # Load the thermal image
    thermal_image = cv2.imread(rf'C:\Users\DELL-DK-STOR\Desktop\Working\4.Validation\1.Fusion\1.source\T-images\DJI_{i}_T.JPG', cv2.IMREAD_COLOR)
    thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)

    # Load the visible RGB image
    visible_image = cv2.imread(rf'C:\Users\DELL-DK-STOR\Desktop\Working\4.Validation\1.Fusion\1.source\V-images\DJI_{i}_V.JPG', cv2.IMREAD_COLOR)

    # Stack the arrays to create a four-band image
    TRGB_image = np.dstack((thermal_image, visible_image))
    RGBT_image = np.dstack((visible_image, thermal_image))

    # Save the multi-band image
    cv2.imwrite(rf'C:\Users\DELL-DK-STOR\Desktop\Working\4.Validation\4.Band-combination\TRGB\TRGB-{i}.TIFF', TRGB_image)
    cv2.imwrite(rf'C:\Users\DELL-DK-STOR\Desktop\Working\4.Validation\4.Band-combination\RGBT\RGBT-{i}.TIFF', RGBT_image)
