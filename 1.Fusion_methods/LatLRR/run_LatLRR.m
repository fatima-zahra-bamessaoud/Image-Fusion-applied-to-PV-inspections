% This is the code for LatLRR algorithm:
% H. Li and X. Wu, ��Infrared and visible image fusion using latent low-rank representation,�� arXiv preprint
% arXiv:1804.08992, 2018.
% 
% The interface is created by the authors of VIFB.

function img = run_latLRR()

    % Hui Li, Xiao-Jun Wu. 
    % Infrared and visible image fusion based on a deep decomposition method.

    % load models
    load('./models/L_8.mat');
    load('./models/L_16.mat');

    % 0 - distinct, 1 - overlapping
    is_overlap = 1;

    % fusion strategy with different norms - l1, nu
    % norm = 'l1';
    norm = 'nu';

    % different project matriices
    % image patch - 8*8, the size of L is 64*64
    L = L_8;
    unit = 8; 
    % image patch - 16*16, the size of L is 256*256
    % L = L_16;
    % unit = 16;
    

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
     img1 = imread(path1);
     img1 = im2double(img1);

     % VI image
     VjpgFileName = VjpgFiles(j).name;
     path2 = fullfile(VFolderPath, VjpgFileName)
     img2 = imread(path2);    
     img2 = im2double(img2);

    % Fusion
    tic

    % jj - the level of decomposition, in our paper is 1 to 4
    for jj=1:4
        str_t = ['L',num2str(unit),'- level ',num2str(jj)];
        disp(str_t);
        de_level = jj;

%         index = 2;
%         path1 = ['./IV_images/IR',num2str(index),'.png'];
%         path2 = ['./IV_images/VIS',num2str(index),'.png'];      
        %fuse_path = ['./fused/fused',num2str(index),'_latlrr_L',num2str(unit),'_level_',num2str(de_level),'_',norm,'_vector_overlap.png'];

        if size(img2, 3) == 1
            F = fusion_method(img1, img2, L, unit, de_level, norm, is_overlap);
        elseif size(img1,3) == 1
            F = zeros(size(img2));
            for i=1:3
                F(:,:,i) = fusion_method(img1, img2(:,:,i), L, unit, de_level, norm, is_overlap);    
            end       
        else
            F= zeros(size(img2));
            for i=1:3
                F(:,:,i) = fusion_method(img1(:,:,i), img2(:,:,i), L, unit, de_level, norm, is_overlap);     
            end    
        end      
        %F = fusion_method(path1, path2, L, unit, de_level, norm, is_overlap);


        % figure;imshow(F);
        %imwrite(F,fuse_path,'png');

    end
    toc
    % Normalize the fused image to the range [0, 255]
     F = double(F);
     F = F - min(F(:));
     F = F / max(F(:));
     F = F * 255;

     % Convert to uint8
     F = uint8(F);

     % Convert to grayscale if it's a multi-band image
     if size(F, 3) == 3
        F = rgb2gray(F);
     end


     fuse_path = ['C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/LatLRR/output/fused-images/lat-fused_',num2str(j+4),'.JPG'];
     imwrite(F,fuse_path,'JPG');
    end
end
