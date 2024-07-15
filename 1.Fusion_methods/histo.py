import os
import glob
import cv2

# Define the folder path of fused images
methods = ["ADF", "CBF", "FPDE", "GLPF", "GTF", "LatLRR", "MGFF", "MSVD", "VGG"]
for method in methods:
    FFolderPath = f'C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/{method}/output/fused-images'
    output_dir = f'C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/{method}/output/fused-images-equalized'

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List all JPG files in the current folder
    FjpgFiles = glob.glob(os.path.join(FFolderPath, '*.JPG'))
    print(f"Processing images of {method} method")

    for k in range(len(FjpgFiles)):
        # Fused image
        FjpgFileName = os.path.basename(FjpgFiles[k])
        path2 = os.path.join(FFolderPath, FjpgFileName)
        image = cv2.imread(path2)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform histogram equalization on the grayscale image
        equalized_image = cv2.equalizeHist(gray_image)

        # Save the equalized image
        output_path = os.path.join(output_dir, f'Equalized_{FjpgFileName}')
        cv2.imwrite(output_path, equalized_image)

        print(f"Image {k} of {method} method processed and saved as {output_path}")

print("Processing complete.")
