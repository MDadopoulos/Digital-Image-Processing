%% Calling readdng doesnt't return double array for unknown reason I couldn't
%find and for that reason I am copied pasted the code from the function to
%run here
%[rawim, XYZ2Cam, wbcoeffs] = readdng('RawImage.dng');

filename='RawImage.dng';
% Read RAW image using LibTIFF library
    obj = Tiff(filename, 'r');
    offsets = getTag(obj, 'SubIFD');
    setSubDirectory(obj, offsets(1));
    rawim = read(obj);
    close(obj);

     % Read useful metadata
    meta_info = imfinfo('RawImage.dng');
    % (x_origin ,y_origin) is the uper left corner of the useful part of the
    %sensor and consequently of the array rawim
    y_origin = meta_info.SubIFDs{1}.ActiveArea(1) + 1;
    x_origin = meta_info.SubIFDs{1}.ActiveArea(2) + 1;
    %width and height of the image (the useful part of array rawim)
    width = meta_info.SubIFDs{1}.DefaultCropSize(1);
    height = meta_info.SubIFDs{1}.DefaultCropSize(2);
    % sensor value corresponding to black
    blacklevel = meta_info.SubIFDs{1}.BlackLevel(1);
    % sensor value corresponding to white
    whitelevel = meta_info.SubIFDs{1}.WhiteLevel;
    wbcoeffs = (meta_info.AsShotNeutral).^-1;
    % green channel will be left unchanged
    wbcoeffs = wbcoeffs/wbcoeffs(2);

    XYZ2Cam = meta_info.ColorMatrix2;
    XYZ2Cam = reshape(XYZ2Cam, 3, 3)';


    % Crop the raw image to the useful part
    rawim = double(rawim(y_origin:(y_origin + height - 1), x_origin:(x_origin + width - 1)));

    % Apply the point transformation
    rawim = (rawim - blacklevel) / double(whitelevel - blacklevel);

    % Clip values outside the range [0, 1]
    rawim = max(0, min(rawim, 1));




%% Set arguments and call dng2rgb function
M=size(rawim,1);
N=size(rawim,2);
method='linear';
bayertype='RGGB';
[Csrgb , Clinear , Cxyz, Ccam] = dng2rgb(rawim , XYZ2Cam , wbcoeffs , bayertype , method , M, N);

% Display all the images calculated from dng2rgb
%% Display the Csrgb image
figure;
imshow(Csrgb);
title("Csrgb");

% Caclulate and display the histograms for each channel
r_channel = Csrgb(:, :, 1);
g_channel = Csrgb(:, :, 2);
b_channel = Csrgb(:, :, 3);

figure;
subplot(3, 1, 1);
histogram(r_channel, 'FaceColor', 'r');
title('Histogram of R channel');

subplot(3, 1, 2);
histogram(g_channel, 'FaceColor', 'g');
title('Histogram of G channel');

subplot(3, 1, 3);
histogram(b_channel, 'FaceColor', 'b');
title('Histogram of B channel');   



%% Display the Clinear image
figure;
imshow(Clinear);
title("Clinear");

% Caclulate and display the histograms for each channel
r_channel = Clinear(:, :, 1);
g_channel = Clinear(:, :, 2);
b_channel = Clinear(:, :, 3);

figure;
subplot(3, 1, 1);
histogram(r_channel, 'FaceColor', 'r');
title('Histogram of R channel');

subplot(3, 1, 2);
histogram(g_channel, 'FaceColor', 'g');
title('Histogram of G channel');

subplot(3, 1, 3);
histogram(b_channel, 'FaceColor', 'b');
title('Histogram of B channel');   


%% Display the Cxyz image
figure;
imshow(Cxyz);
title("Cxyz");

% Caclulate and display the histograms for each channel
r_channel = Cxyz(:, :, 1);
g_channel = Cxyz(:, :, 2);
b_channel = Cxyz(:, :, 3);

figure;
subplot(3, 1, 1);
histogram(r_channel, 'FaceColor', 'r');
title('Histogram of R channel');

subplot(3, 1, 2);
histogram(g_channel, 'FaceColor', 'g');
title('Histogram of G channel');

subplot(3, 1, 3);
histogram(b_channel, 'FaceColor', 'b');
title('Histogram of B channel');   




%% Display the Ccam image
figure;
imshow(Ccam);
title("Ccam");

% Caclulate and display the histograms for each channel
r_channel = Ccam(:, :, 1);
g_channel = Ccam(:, :, 2);
b_channel = Ccam(:, :, 3);

figure;
subplot(3, 1, 1);
histogram(r_channel, 'FaceColor', 'r');
title('Histogram of R channel');

subplot(3, 1, 2);
histogram(g_channel, 'FaceColor', 'g');
title('Histogram of G channel');

subplot(3, 1, 3);
histogram(b_channel, 'FaceColor', 'b');
title('Histogram of B channel');   
