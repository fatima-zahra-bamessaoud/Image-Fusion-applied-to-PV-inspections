function img = run_MSVD()

    addpath(genpath(cd));

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
        path1 = fullfile(TFolderPath, TjpgFileName);
        I = double(imread(path1))/255;

        % VI image
        VjpgFileName = VjpgFiles(j).name;
        path2 = fullfile(VFolderPath, VjpgFileName);
        V = double(imread(path2))/255;

        I = double(I);
        V = double(V);

        calc_metric = 0; % Calculate the metrics is time consuming, it is used for quantitative evaluation. Set it to 0 if you do not want to do it.

        tic;
        if size(V,3) == 1   % for gray images       
            % % Pixel-and region-based image fusion with complex wavelets(2007)
            [M, N] = size(I);
            I4 = imresize(I, [M + mod(M, 2) N + mod(N, 2)]);
            V4 = imresize(V, [M + mod(M, 2) N + mod(N, 2)]);

            % Image Fusion technique using Multi-resolution singular Value decomposition (2011)
            % apply MSVD
            tic;
            [Y1, U1] = MSVD(I4);
            [Y2, U2] = MSVD(V4);

            % fusion starts
            X6.LL = 0.5 * (Y1.LL + Y2.LL);

            D = (abs(Y1.LH) - abs(Y2.LH)) >= 0; 
            X6.LH = D .* Y1.LH + (~D) .* Y2.LH;
            D = (abs(Y1.HL) - abs(Y2.HL)) >= 0; 
            X6.HL = D .* Y1.HL + (~D) .* Y2.HL;
            D = (abs(Y1.HH) - abs(Y2.HH)) >= 0; 
            X6.HH = D .* Y1.HH + (~D) .* Y2.HH;

            U = 0.5 * (U1 + U2);

            % apply IMSVD
            X6 = IMSVD(X6, U);
            toc;

            X6 = mat2gray(X6);
            imgf = X6;

        elseif size(I,3) == 1 
            imgf = zeros(size(V)); 
            M = size(I, 1);
            N = size(I, 2);
            for i = 1:3
                I4 = imresize(I, [M + mod(M, 2) N + mod(N, 2)]);
                V4 = imresize(V(:,:,i), [M + mod(M, 2) N + mod(N, 2)]);

                [Y1, U1] = MSVD(I4);
                [Y2, U2] = MSVD(V4);

                % Initialize X6{i} as a struct
                X6{i} = struct('LL', [], 'LH', [], 'HL', [], 'HH', []);

                % fusion starts
                X6{i}.LL = 0.5 * (Y1.LL + Y2.LL);

                D = (abs(Y1.LH) - abs(Y2.LH)) >= 0; 
                X6{i}.LH = D .* Y1.LH + (~D) .* Y2.LH;
                D = (abs(Y1.HL) - abs(Y2.HL)) >= 0; 
                X6{i}.HL = D .* Y1.HL + (~D) .* Y2.HL;
                D = (abs(Y1.HH) - abs(Y2.HH)) >= 0; 
                X6{i}.HH = D .* Y1.HH + (~D) .* Y2.HH;

                U = 0.5 * (U1 + U2);

                % apply IMSVD
                X6{i} = IMSVD(X6{i}, U);

                imgf(:,:,i) = X6{i};
            end 

        else
            imgf = zeros(size(I)); 
            M = size(I, 1);
            N = size(I, 2);
            for i = 1:3
                I4 = imresize(I(:,:,i), [M + mod(M, 2) N + mod(N, 2)]);
                V4 = imresize(V(:,:,i), [M + mod(M, 2) N + mod(N, 2)]);

                [Y1, U1] = MSVD(I4);
                [Y2, U2] = MSVD(V4);

                % Initialize X6{i} as a struct
                X6{i} = struct('LL', [], 'LH', [], 'HL', [], 'HH', []);

                % fusion starts
                X6{i}.LL = 0.5 * (Y1.LL + Y2.LL);

                D = (abs(Y1.LH) - abs(Y2.LH)) >= 0; 
                X6{i}.LH = D .* Y1.LH + (~D) .* Y2.LH;
                D = (abs(Y1.HL) - abs(Y2.HL)) >= 0; 
                X6{i}.HL = D .* Y1.HL + (~D) .* Y2.HL;
                D = (abs(Y1.HH) - abs(Y2.HH)) >= 0; 
                X6{i}.HH = D .* Y1.HH + (~D) .* Y2.HH;

                U = 0.5 * (U1 + U2);

                % apply IMSVD
                X6{i} = IMSVD(X6{i}, U);
                imgf(:,:,i) = X6{i};
            end
        end
        toc;
    
    
         % Convert to grayscale if it's a multi-band image
        if size(imgf, 3) == 3
            img = rgb2gray(imgf);
        end

        img = uint8(255 * (img - min(img(:))) / (max(img(:)) - min(img(:))));

        fuse_path = ['C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/MSVD/output/fused-images/MSVD-fused_', num2str(j + 4), '.JPG'];
        imwrite(img, fuse_path, 'JPG');
    end
end
