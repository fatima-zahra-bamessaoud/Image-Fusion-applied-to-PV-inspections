% This code is the implementation of "Multi-scale Guided Image and Video Fusion: A Fast and Efficient Approach" 
% Cite this article as:
% Bavirisetti, D.P., Xiao, G., Zhao, J. et al. Circuits Syst Signal Process (2019).
%https://doi.org/10.1007/s00034-019-01131-z
% 
% The interface is created by the authors of VIFB.
 
function run_MGFF()
    % Guided image filter parameters
    r = 9; eps = 10^3;

    % Paths
     % Define the folder path of thermal images
     TFolderPath = 'C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/1.source/T-alignedF';
     % List all JPG files in the current folder
     TjpgFiles = dir(fullfile(TFolderPath, '*.JPG'));

     % Define the folder path of thermal images
     VFolderPath = 'C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/1.source/V-alignedF';
     % List all JPG files in the current folder
     VjpgFiles = dir(fullfile(VFolderPath, '*.JPG'));

     % IR image
     for j = 1:length(TjpgFiles)
     TjpgFileName = TjpgFiles(j).name;
     path1 = fullfile(TFolderPath, TjpgFileName)
     I1 = double(imread(path1));
        if size(I1, 3) == 1
            I1 = repmat(I1, [1, 1, 3]);
        end

        % VI image
        VjpgFileName = VjpgFiles(j).name;
        path2 = fullfile(VFolderPath, VjpgFileName)
        I2 = double(imread(path2));    

        % Apply multi-scale guided image fusion on source images
        F = fuse_MGF_RGB(I1, I2, r, eps);

        % Convert to grayscale if it's a multi-band image
        if size(F, 3) == 3
            F = rgb2gray(F);
        end

        % Save the output image
        fuse_path = ['C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/MGFF/output/fused-images/MGFF-fused_',num2str(j+4),'.JPG'];
        imwrite(uint8(F), fuse_path, 'JPG');

    end
end
