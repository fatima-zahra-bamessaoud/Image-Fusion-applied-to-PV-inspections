% The main code of CBF is provided by the authors of CBF (see following information).
% The interface is created by the authors of VIFB. Necessary modifications
% are also made to adapt to VIFB.

%%% Image Fusion using Details computed from Cross Bilteral Filter output.
%%% Details are obtained by subtracting original image by cross bilateral filter output.
%%% These details are used to find weights (Edge Strength) for fusing the images.
%%% Author : B. K. SHREYAMSHA KUMAR 

%%% Copyright (c) 2013 B. K. Shreyamsha Kumar 
%%% All rights reserved.

%%% Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, 
%%% modify, and distribute this code (the source files) and its documentation for any purpose, provided that the 
%%% copyright notice in its entirety appear in all copies of this code, and the original source of this code, 
%%% This should be acknowledged in any publication that reports research using this code. The research is to be 
%%% cited in the bibliography as:

%%% B. K. Shreyamsha Kumar, Image fusion based on pixel significance using cross bilateral filter", Signal, Image 
%%% and Video Processing, pp. 1-12, 2013. (doi: 10.1007/s11760-013-0556-9)

function img = run_CBF()

    %%% Fusion Method Parameters.
    cov_wsize=5;
    visualization=0;
    %%% Bilateral Filter Parameters.
    sigmas=1.8;  %%% Spatial (Geometric) Sigma. 1.8
    sigmar=25; %%% Range (Photometric/Radiometric) Sigma.25 256/10
    ksize=11;   %%% Kernal Size  (should be odd).

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
     x{1} = imread(path1);

     % VI image
     VjpgFileName = VjpgFiles(j).name;
     path2 = fullfile(VFolderPath, VjpgFileName)
     x{2} = imread(path2);
     

    if visualization == 1
        figure, imshow((uint8(x{1})));
    end

    if visualization == 1
        figure, imshow(uint8(x{2}));
    end
         
    [M]=size(x{1},1);
    [N]=size(x{1},2);
    
    x{1}= double(x{1});
    x{2}= double(x{2});

    %%% Cross Bilateral Filter.
    tic
    if size(x{2},3)==1   %for gray images    
        cbf_out{1}=cross_bilateral_filt2Df(x{1},x{2},sigmas,sigmar,ksize);
        detail{1}=double(x{1})-cbf_out{1};
        cbf_out{2}= cross_bilateral_filt2Df(x{2},x{1},sigmas,sigmar,ksize);
        detail{2}=double(x{2})-cbf_out{2};

        %%% Fusion Rule (IEEE Conf 2011).
        xfused=cbf_ieeeconf2011f(x,detail,cov_wsize);
    elseif size(x{1},3)==1                  %for color images
        xfused=zeros(size(x{1}));
        cbf_out{1}=zeros(size(x{1}));
        cbf_out{2}=zeros(size(x{1}));
        detail{1}=zeros(size(x{1}));
        detail{2}=zeros(size(x{1}));
            
        x_new{1} = zeros(size(x{1},1),size(x{1},2));
        x_new{2} = zeros(size(x{1},1),size(x{1},2));
        detail_new{1} = zeros(size(x{1},1),size(x{1},2));
        detail_new{2} = zeros(size(x{1},1),size(x{1},2));
        
        for i=1:3
            
            cbf_out{1}(:,:,i)=cross_bilateral_filt2Df(x{1},x{2}(:,:,i),sigmas,sigmar,ksize);
            detail{1}(:,:,i)=double(x{1})-cbf_out{1}(:,:,i);
            cbf_out{2}(:,:,i)= cross_bilateral_filt2Df(x{2}(:,:,i),x{1},sigmas,sigmar,ksize);
            detail{2}(:,:,i)=double(x{2}(:,:,i))-cbf_out{2}(:,:,i);

%             %%% Fusion Rule (IEEE Conf 2011).
%             xfused(:,:,i)=cbf_ieeeconf2011f(x(:,:,i),detail(:,:,i),cov_wsize);
            
            x_new{1}(:,:) = x{1};
            x_new{2}(:,:) = x{2}(:,:,i);
            detail_new{1}(:,:) = detail{1}(:,:,i);
            detail_new{2}(:,:) = detail{2}(:,:,i);
            xfused(:,:,i)=cbf_ieeeconf2011f(x_new,detail_new,cov_wsize);
        end      
        
    else                    
        xfused=zeros(size(x{1}));
        cbf_out{1}=zeros(size(x{1}));
        cbf_out{2}=zeros(size(x{1}));
        detail{1}=zeros(size(x{1}));
        detail{2}=zeros(size(x{1}));
            
        x_new{1} = zeros(size(x{1},1),size(x{1},2));
        x_new{2} = zeros(size(x{1},1),size(x{1},2));
        detail_new{1} = zeros(size(x{1},1),size(x{1},2));
        detail_new{2} = zeros(size(x{1},1),size(x{1},2));
        
        for i=1:3
            
            cbf_out{1}(:,:,i)=cross_bilateral_filt2Df(x{1}(:,:,i),x{2}(:,:,i),sigmas,sigmar,ksize);
            detail{1}(:,:,i)=double(x{1}(:,:,i))-cbf_out{1}(:,:,i);
            cbf_out{2}(:,:,i)= cross_bilateral_filt2Df(x{2}(:,:,i),x{1}(:,:,i),sigmas,sigmar,ksize);
            detail{2}(:,:,i)=double(x{2}(:,:,i))-cbf_out{2}(:,:,i);

%             %%% Fusion Rule (IEEE Conf 2011).
%             xfused(:,:,i)=cbf_ieeeconf2011f(x(:,:,i),detail(:,:,i),cov_wsize);
            
            x_new{1}(:,:) = x{1}(:,:,i);
            x_new{2}(:,:) = x{2}(:,:,i);
            detail_new{1}(:,:) = detail{1}(:,:,i);
            detail_new{2}(:,:) = detail{2}(:,:,i);
            xfused(:,:,i)=cbf_ieeeconf2011f(x_new,detail_new,cov_wsize);
        end
%             %%% Fusion Rule (IEEE Conf 2011).
%             xfused=cbf_ieeeconf2011f(x,detail,cov_wsize);
    end 
    toc

    % Normalize the fused image to the range [0, 255]
     xfused = double(xfused);
     xfused = xfused - min(xfused(:));
     xfused = xfused / max(xfused(:));
     xfused = xfused * 255;

     % Convert to uint8
     xfused = uint8(xfused);

     % Convert to grayscale if it's a multi-band image
     if size(xfused, 3) == 3
        xfused = rgb2gray(xfused);
     end


     fuse_path = ['C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/CBF/output/fused-images/CBF-fused_',num2str(j+4),'.JPG'];
     imwrite(xfused,fuse_path,'JPG');
    
   if visualization == 1
        figure, imshow((xfused), [])
   end
    end
end
