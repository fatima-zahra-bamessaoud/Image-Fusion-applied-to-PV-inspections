% Clear all variables and close all figures
clear all;
close all;
clc;

% Paths
% Define the folder path of thermal images
TFolderPath = 'C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/1.source/T-alignedF';
% List all JPG files in the current folder
TjpgFiles = dir(fullfile(TFolderPath, '*.JPG'));

% Define the folder path of visible images
VFolderPath = 'C:/Users/DELL-DK-STOR/Desktop/Working/1.Fusion_methods/1.source/V-alignedF';
% List all JPG files in the current folder
VjpgFiles = dir(fullfile(VFolderPath, '*.JPG'));

% Define the folder path of fused images
methods = ["ADF", "CBF", "FPDE", "GLPF", "GTF", "LatLRR", "MGFF", "MSVD", "VGG"];
for method = methods
    FFolderPath = fullfile('C:', 'Users', 'DELL-DK-STOR', 'Desktop', 'Working', '1.Fusion_methods', char(method), 'output', 'fused-images');
    % List all JPG files in the current folder
    FjpgFiles = dir(fullfile(FFolderPath, '*.JPG'));
    fprintf('Processing images of %s method\n', method);

    for j = 1:length(VjpgFiles)
        % Fused image
        FjpgFileName = FjpgFiles(j).name;
        path3 = fullfile(FFolderPath, FjpgFileName);
        fused_image = imread(path3);

        % IR image
        TjpgFileName = TjpgFiles(j).name;
        path1 = fullfile(TFolderPath, TjpgFileName);
        T_image = imread(path1);

        % VI image
        VjpgFileName = VjpgFiles(j).name;
        path2 = fullfile(VFolderPath, VjpgFileName);
        V_image = imread(path2);

        % Convert to grayscale if necessary
        if size(T_image, 3) == 3
            T_image = rgb2gray(T_image);
        end
        if size(V_image, 3) == 3
            V_image = rgb2gray(V_image);
        end
        if size(fused_image, 3) == 3
            fused_image = rgb2gray(fused_image);
        end
        
        % Define the vertices of the polygonal ROI
        x_vertices = [1, 640, 640, 1];
        y_vertices = [140, 160, 260, 240];

        % Create the ROI mask
        roi_mask = poly2mask(x_vertices, y_vertices, size(fused_image, 1), size(fused_image, 2));
        
        % Extract the ROI using the mask
        T_roi = T_image .* uint8(roi_mask);
        V_roi = V_image .* uint8(roi_mask);
        roi = fused_image .* uint8(roi_mask);
        
        % Display the image with the ROI
        %imshow(roi);
        %title('ROI');

        fprintf('Image: %d\n', j + 4);
        
        % PSNR
        psnr_value = evaluate_fusion(T_roi, roi);
        fprintf('PSNR: %.4f dB\n', psnr_value);

        % RMSE
        rmse_value = metricsRmse(T_roi, V_roi, roi);
        fprintf('RMSE: %.4f\n', rmse_value);

        % Entropy
        entropy_value = metricsEntropy(T_roi, V_roi, roi);
        IR_entropy_value = metricsEntropy(T_roi, V_roi, T_roi);
        fprintf('Entropy Ratio: %.4f\n', entropy_value./IR_entropy_value);

        % Mutual information
        MI_value = metricsMutinf(T_roi, V_roi, roi);
        fprintf('Mutual Info: %.4f\n', MI_value);

        % Spatial frequency
        SF_value = metricsSpatial_frequency(T_roi, V_roi, roi);
        fprintf('Spatial Frequency: %.4f\n', SF_value);

        % Variance
        Var_value  = metrics_Variance(T_roi, V_roi, roi);
        fprintf('Variance: %.4f\n', Var_value);

        % SSIM
        % Normalize the images
        T_roi = mat2gray(T_roi);
        V_roi = mat2gray(V_roi);
        roi = mat2gray(roi);
        Mf_value = metric_ssim(roi, T_roi);
        fprintf('SSIM: %.4f\n', Mf_value);
    end
end

% Evaluation metric functions
% PSNR
function psnr_value = evaluate_fusion(T_image, fused_image)
    psnr_value = metric_PSNR(fused_image, T_image);
end

function thePSNR = metric_PSNR(I1, I2)
    I1 = double(I1);
    I2 = double(I2);
    M = max(max(I1(:)), max(I2(:)));
    mse = mean((I1(:) - I2(:)).^2);
    thePSNR = 10 * log10(M^2 / mse);
end

% RMSE
function res = metricsRmse(T_image, V_image, fused_image)
    img1 = double(T_image);
    img2 = double(V_image);
    fused = double(fused_image);
    % Resize images to the smallest dimensions
    [m, n, ~] = size(fused);
    img1 = imresize(img1, [m, n]);
    img2 = imresize(img2, [m, n]);
    
    % Get the number of channels in fused image
    [~, ~, b] = size(fused);

    % Initialize the result
    g = zeros(1, b);

    % Check if images are single-channel or multi-channel
    if b == 1
        g = Rmse(img1, img2, fused);
        res = g;
    else
        for k = 1:b
            if size(img2, 3) == 1
                g(k) = Rmse(img1(:,:,k), img2, fused(:,:,k));
            else
                g(k) = Rmse(img1(:,:,k), img2(:,:,k), fused(:,:,k));
            end
        end
        res = mean(g);
    end
end

function output = Rmse(T_image, V_image, fused_image)
    img1 = T_image;
    img2 = V_image;
    fused = fused_image;
    % Compute RMSE for each image pair
    rmseVF = mse(img1, fused);
    rmseIF = mse(img2, fused);

    % Average the RMSE values
    rmse = rmseVF + rmseIF;
    output = rmse / 2.0;
end

function res0 = mse(a, b)
    % Ensure images are grayscale
    if size(a, 3) > 1
        a = rgb2gray(a);
    end

    if size(b, 3) > 1
        b = rgb2gray(b);
    end

    % Get the size of the images
    [m, n] = size(a);

    % Ensure the sizes match
    assert(all(size(a) == size(b)), 'Images must be the same size.');

    % Calculate mean squared error
    temp = sqrt(sum(sum((a - b).^2)));
    res0 = temp / (m * n);
end

% Entropy
function res = metricsEntropy(img1, img2, fused)
    fused = double(fused); 
    [m,n,b] = size(fused); 
    [m1,n1,b1] = size(img2);

    if b == 1
        g = Entropy(img1,img2,fused);
        res = g;
    elseif b1 == 1
        for k = 1 : b 
           g(k) = Entropy(img1(:,:,k),img2,fused(:,:,k)); 
        end 
        res = mean(g); 
    else
        for k = 1 : b 
            g(k) = Entropy(img1(:,:,k),img2(:,:,k),fused(:,:,k)); 
        end 
        res = mean(g); 
    end
end

function output = Entropy(img1, img2,fused) 
    h = fused;
    s = size(size(h));
    if s(2) == 3 
        h1 = rgb2gray(h);
    else
        h1 = h;
    end    
    h1 = double(h1);
    [m,n] = size(h1);
    X = zeros(1,256);
    result = 0;
    P = zeros(1, 256); % Initialize P array
    for i = 1:m
        for j = 1:n
            X(floor(h1(i,j))+1) = X(floor(h1(i,j))+1) + 1;
        end
    end
    for k = 1:256
        P(k) = X(k) / (m * n);
        if (P(k) ~= 0)
            result = P(k) * log2(P(k)) + result;
        end
    end
    result = -result;
    EN1 = result;
    output = EN1;
end

% Mutual information
function res = metricsMutinf(img1,img2,fused)
    fused = double(fused); 
    % Get the size of img 
    [m,n,b] = size(fused); 
    [m1,n1,b1] = size(img2);

    if b == 1
        g = Mutinf_single(img1,img2,fused);
        res = g;
    elseif b1 == 1
        for k = 1 : b 
           g(k) = Mutinf_single(img1(:,:,k), img2,fused(:,:,k)); 
        end 
        res = mean(g); 
    else
        for k = 1 : b 
            g(k) = Mutinf_single(img1(:,:,k), img2(:,:,k),fused(:,:,k)); 
        end 
        res = mean(g); 
    end
end

function output = Mutinf_single(img1,img2,fused)
  miVI = mutinf(img1, fused);
  miIR = mutinf(img2, fused);
  mi = miVI + miIR;
  output = mi;
end

function mi = mutinf(a, b)
    if size(a,3) > 1
        a=rgb2gray(a);
    end
    
    if size(b,3) > 1
        b=rgb2gray(b);
    end

    a=double(a);
    b=double(b);
    [Ma,Na] = size(a);
    [Mb,Nb] = size(b);
    M=min(Ma,Mb);
    N=min(Na,Nb);

    % Initialize histogram arrays
    hab = zeros(256,256);
    ha = zeros(1,256);
    hb = zeros(1,256);

    % Normalize
    if max(max(a))~=min(min(a))
        a = (a-min(min(a)))/(max(max(a))-min(min(a)));
    else
        a = zeros(M,N);
    end

    if max(max(b))-min(min(b))
        b = (b-min(min(b)))/(max(max(b))-min(min(b)));
    else
        b = zeros(M,N);
    end

    a = double(int16(a*255))+1;
    b = double(int16(b*255))+1;

    % Compute histograms
    for i=1:M
        for j=1:N
           indexx =  a(i,j);
           indexy = b(i,j) ;
           hab(indexx,indexy) = hab(indexx,indexy)+1;% Joint histogram
           ha(indexx) = ha(indexx)+1;% a image histogram
           hb(indexy) = hb(indexy)+1;% b image histogram
       end
    end

    % Calculate joint entropy
    hsum = sum(sum(hab));
    index = find(hab~=0);
    p = hab/hsum;
    Hab = sum(sum(-p(index).*log(p(index))));

    % Calculate a image entropy
    hsum = sum(sum(ha));
    index = find(ha~=0);
    p = ha/hsum;
    Ha = sum(sum(-p(index).*log(p(index))));

    % Calculate b image entropy
    hsum = sum(sum(hb));
    index = find(hb~=0);
    p = hb/hsum;
    Hb = sum(sum(-p(index).*log(p(index))));

    % Calculate mutual information between a and b
    mi = Ha+Hb-Hab;

    % Calculate normalized mutual information
    mi1 = hab/(Ha+Hb); 
end

% Spatial frequency
function res = metricsSpatial_frequency(img1, img2, fused)
    fused=double(fused);
    [m,n]=size(fused);
    RF=0;
    CF=0;

    for fi=1:m
        for fj=2:n
            RF=RF+(fused(fi,fj)-fused(fi,fj-1)).^2;
        end
    end

    RF=RF/(m*n);

    for fj=1:n
        for fi=2:m
            CF=CF+(fused(fi,fj)-fused(fi-1,fj)).^2;
        end
    end

    CF=CF/(m*n);

    res=sqrt(RF+CF);
end

% Variance
function res  = metrics_Variance(img1,img2,fused)
    fused = double(fused); 
    [m,n,b] = size(fused); 
    [m1,n1,b1] = size(img2);

    if b == 1
        [a,g] = Variance(img1,img2,fused);
        img_var = g;
    elseif b1 == 1
        for k = 1 : b 
           [a,g(k)] = Variance(img1(:,:,k), img2,fused(:,:,k)); 
        end 
        img_var = mean(g); 
    else
        for k = 1 : b 
           [a,g(k)] = Variance(img1(:,:,k), img2(:,:,k),fused(:,:,k)); 
        end 
        img_var = mean(g); 
    end
    res = img_var;
end

function [img_mean,img_var] = Variance(img1,img2,fused)
    if size(fused,3) > 1 
        fused=rgb2gray(fused);  
    end
    fused = double(fused); 
    [r, c] = size(fused); 

    % Mean value 
    img_mean = mean(mean(fused)); 

    % Variance 
    img_var = sqrt(sum(sum((fused - img_mean).^2)) / (r * c ));
end

% SSIM
function theSSIM = metric_ssim(I1, I2)
    % Ensure the images are of type double
    I1 = double(I1);
    I2 = double(I2);

    % Parameters for SSIM
    K1 = 0.01;
    K2 = 0.03;
    L = 255; % Dynamic range of pixel values
    C1 = (K1 * L)^2;
    C2 = (K2 * L)^2;

    % Gaussian filter
    N = 11;
    sigma = 1.5;
    [X, Y] = meshgrid(-(N-1)/2:(N-1)/2, -(N-1)/2:(N-1)/2);
    gaussian = exp(-(X.^2 + Y.^2) / (2*sigma^2));
    gaussian = gaussian / sum(gaussian(:));

    % Windowed means
    mu1 = filter2(gaussian, I1, 'valid');
    mu2 = filter2(gaussian, I2, 'valid');
    mu1_mu2 = mu1 .* mu2;
    mu1_sq = mu1.^2;
    mu2_sq = mu2.^2;

    % Windowed variances and covariances
    sigma1_sq = filter2(gaussian, I1.^2, 'valid') - mu1_sq;
    sigma2_sq = filter2(gaussian, I2.^2, 'valid') - mu2_sq;
    sigma12 = filter2(gaussian, I1.*I2, 'valid') - mu1_mu2;

    % SSIM calculation
    numerator = (2 * mu1_mu2 + C1) .* (2 * sigma12 + C2);
    denominator = (mu1_sq + mu2_sq + C1) .* (sigma1_sq + sigma2_sq + C2);
    ssim_map = numerator ./ denominator;
    theSSIM = mean(ssim_map(:));
end
