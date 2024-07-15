% This is the program of ADF from the paper:
%
% D. P. Bavirisetti and R. Dhuli, ��Fusion of infrared and visible
% sensor images based on anisotropic diffusion and karhunenloeve transform,�� IEEE Sensors Journal, vol. 16, no. 1, pp.
% 203�C209, 2016.
%
% The function ADF() is provided by the authors of ADF.
% The interface is created by the authors of VIFB.


function img = run_ADF()
     % Paths
     % Define the folder path of thermal images
     TFolderPath = 'C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/1.source/T-alignedF';
     % List all JPG files in the current folder
     TjpgFiles = dir(fullfile(TFolderPath, '*.JPG'));

     % Define the folder path of visible images
     VFolderPath = 'C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/1.source/V-alignedF';
     % List all JPG files in the current folder
     VjpgFiles = dir(fullfile(VFolderPath, '*.JPG'));

     % IR image
     for j = 1:length(TjpgFiles)
     TjpgFileName = TjpgFiles(j).name;
     path1 = fullfile(TFolderPath, TjpgFileName)
     I1 = imread(path1);

     % VI image
     VjpgFileName = VjpgFiles(j).name;
     path2 = fullfile(VFolderPath, VjpgFileName)
     I2 = imread(path2);
     
     tic;
     if size(I2, 3) == 1
         fuseimage = ADF(I1, I2);
     elseif size(I1,3) == 1
         fuseimage = zeros(size(I2));
         for i=1:3
            fuseimage(:,:,i) = ADF(I1,I2(:,:,i));    
         end       
     else
         fuseimage = zeros(size(I2));
         for i=1:3
            fuseimage(:,:,i) = ADF(I1(:,:,i),I2(:,:,i));    
         end    
     end
     toc;   
     
     % Normalize the fused image to the range [0, 255]
     fuseimage = double(fuseimage);
     fuseimage = fuseimage - min(fuseimage(:));
     fuseimage = fuseimage / max(fuseimage(:));
     fuseimage = fuseimage * 255;

     % Convert to uint8
     img = uint8(fuseimage);

     % Convert to grayscale if it's a multi-band image
     if size(img, 3) == 3
        img = rgb2gray(img);
     end


     fuse_path = ['C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/ADF/output/fused-images/ADF-fused_',num2str(j+4),'.JPG'];
     imwrite(img,fuse_path,'JPG');
     end
end

function res = ADF(I1, I2)
     %ANISOTROPIC DIFFUSION
     num_iter = 10;
     delta_t = 0.15;
     kappa = 30;
     option = 1;

     A1 = anisodiff2D(I1,num_iter,delta_t,kappa,option);
     A2= anisodiff2D(I2,num_iter,delta_t,kappa,option);

     D1=double(I1)-A1;
     D2=double(I2)-A2;

     C1 = cov([D1(:) D2(:)]);
     [V11, D11] = eig(C1);
     if D11(1,1) >= D11(2,2)
        pca1 = V11(:,1)./sum(V11(:,1));
     else  
        pca1 = V11(:,2)./sum(V11(:,2));
     end

     imf1 = pca1(1)*D1 + pca1(2)*D2;
     imf2=(0.5*A1+0.5*A2);

     res=(double(imf1)+double(imf2));
end
 