% This is the code for GTF algorithm:
% J. Ma, C. Chen, C. Li, and J. Huang, ��Infrared and visible image fusion via gradient transfer 
% and total variation minimization,�� Information Fusion, vol. 31, pp. 100�C109, 2016.
%
% The code of GTF is provided by the authors of GTF.
% The interface is created by the authors of VIFB.

function img = run_GTF()

    addpath(genpath(cd));

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
     I = double(imread(path1))/255;

     % VI image
     VjpgFileName = VjpgFiles(j).name;
     path2 = fullfile(VFolderPath, VjpgFileName)
     V = double(imread(path2))/255;

    nmpdef;
    pars_irn = irntvInputPars('l1tv');

    pars_irn.adapt_epsR   = 1;
    pars_irn.epsR_cutoff  = 0.01;   % This is the percentage cutoff
    pars_irn.adapt_epsF   = 1;
    pars_irn.epsF_cutoff  = 0.05;   % This is the percentage cutoff
    pars_irn.pcgtol_ini = 1e-4;
    pars_irn.loops      = 5;
    pars_irn.U0         = I-V;
    pars_irn.variant       = NMP_TV_SUBSTITUTION;
    pars_irn.weight_scheme = NMP_WEIGHTS_THRESHOLD;
    pars_irn.pcgtol_ini    = 1e-2;
    pars_irn.adaptPCGtol   = 1;

    tic;
    U = irntv(I-V, {}, 4, pars_irn);
    toc;

    X=U+V;
    
    % Normalize the fused image to the range [0, 255]
     X = double(X);
     X = X - min(X(:));
     X = X / max(X(:));
     X = X * 255;

     % Convert to uint8
     X = uint8(X);

     % Convert to grayscale if it's a multi-band image
     if size(X, 3) == 3
        X = rgb2gray(X);
     end


     fuse_path = ['C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/GTF/output/fused-images/GTF-fused_',num2str(j+4),'.JPG'];
     imwrite(X,fuse_path,'JPG');
    end
end
