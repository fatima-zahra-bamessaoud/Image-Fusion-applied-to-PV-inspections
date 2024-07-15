import subprocess
import os
import glob

# Define the folder path of thermal images
TFolderPath = r'C:\Users\DELL-DK-STOR\Desktop\Working\4.Validation\1.Fusion\1.source\T-images'
# List all JPG files in the current folder
TjpgFiles = glob.glob(os.path.join(TFolderPath, '*.JPG'))

# Define the folder path of fused images
methods = ["GTF", "MGFF"]
for method in methods:
    FFolderPath = rf'C:\Users\DELL-DK-STOR\Desktop\Working\4.Validation\1.Fusion\{method}\output'
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
