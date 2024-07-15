% This is the program of FPDE: 
% D. P. Bavirisetti, G. Xiao, and G. Liu, ��Multi-sensor image
% fusion based on fourth order partial differential equations,�� in
% 2017 20th International Conference on Information Fusion
% (Fusion). IEEE, 2017, pp. 1�C9.
% Codes are provided by the authors of FPDE.
%
% The interface is created by the authors of VIFB. Necessary modifications
% are made to be integrated into VIFB. 

function img = run_FPDE()

    visualization=0;
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
     I1 = imread(path1);

     % VI image
     VjpgFileName = VjpgFiles(j).name;
     path2 = fullfile(VFolderPath, VjpgFileName)
     I2 = imread(path2);
    

    if visualization == 1
        figure, imshow((uint8(I1)));
    end

    if visualization == 1
        figure, imshow(uint8(I2));
    end

    tic;
    if size(I2, 3) == 1
        fuseimage = FPDE(I1, I2);
    elseif size(I1,3) == 1
        fuseimage = zeros(size(I2));
        for i=1:3
            fuseimage(:,:,i) = FPDE(I1,I2(:,:,i));    
        end       
    else
        fuseimage = zeros(size(I2));
        for i=1:3
           fuseimage(:,:,i) = FPDE(I1(:,:,i),I2(:,:,i));    
        end    
    end   
    toc;
    
    % Normalize the fused image to the range [0, 255]
     fuseimage = double(fuseimage);
     fuseimage = fuseimage - min(fuseimage(:));
     fuseimage = fuseimage / max(fuseimage(:));
     fuseimage = fuseimage * 255;

     % Convert to uint8
     fuseimage = uint8(fuseimage);

     % Convert to grayscale if it's a multi-band image
     if size(fuseimage, 3) == 3
        fuseimage = rgb2gray(fuseimage);
     end


     fuse_path = ['C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/FPDE/output/fused-images/FPDE-fused_',num2str(j+4),'.JPG'];
     imwrite(fuseimage,fuse_path,'JPG');
    
   if visualization == 1
        figure, imshow((fuseimage), [])
   end
    
    end
end
    
function res = FPDE (I1,I2)

    % Assigning values to the parameters
    n=15; 
    dt=0.9;
    k=4;

    % Decomposing input images as base and detail layers

    [A1]=fpdepyou(I1,n);
    [A2]=fpdepyou(I2,n);
    D1=double(I1)-double(A1);
    D2=double(I2)-double(A2);

    A(:,:,1)=A1;
    A(:,:,2)=A2;

    D(:,:,1)=D1;
    D(:,:,2)=D2;

    % Detail layer fusion 

    C1 = cov([D1(:) D2(:)]);
    [V11, D11] = eig(C1);
    if D11(1,1) >= D11(2,2)
        pca1 = V11(:,1)./sum(V11(:,1));
    else  
        pca1 = V11(:,2)./sum(V11(:,2));
    end
    imf1 = pca1(1)*D1 + pca1(2)*D2;

    % Base layer fusion 
    imf2=(0.5*A1+0.5*A2);

    % Final fused image
    res=(double(imf1)+double(imf2));
end
