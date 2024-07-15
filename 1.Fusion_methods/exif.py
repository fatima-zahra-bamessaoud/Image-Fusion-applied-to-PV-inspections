import subprocess
import os
import glob

# Define the folder path of thermal images
TFolderPath = r'C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/1.source/T-alignedF'
# List all JPG files in the current folder
TjpgFiles = glob.glob(os.path.join(TFolderPath, '*.JPG'))

# Define the folder path of fused images
methods = ["ADF", "CBF", "FPDE", "GLPF", "GTF", "LatLRR", "MGFF", "MSVD", "VGG"]
for method in methods:
    FFolderPath = f'C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/{method}/output/fused-images'
    # List all JPG files in the current folder
    FjpgFiles = glob.glob(os.path.join(FFolderPath, '*.JPG'))
    print(f"Processing images of {method} method")

    for k in range(len(TjpgFiles)):
        # IR image
        TjpgFileName = os.path.basename(TjpgFiles[k])
        path1 = os.path.join(TFolderPath, TjpgFileName)

        # Fused image
        FjpgFileName = os.path.basename(FjpgFiles[k])
        path2 = os.path.join(FFolderPath, FjpgFileName)
        
        # Copy Exif data from original thermal image to the fused image using ExifTool
        subprocess.run([r'C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/exiftool.exe', '-tagsFromFile', path1, path2])
        
        print(f"Image {k} is processed")

print("Processing complete.")
