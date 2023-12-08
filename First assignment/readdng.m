function [rawim, XYZ2Cam, wbcoeffs] = readdng(filename)
    % Read RAW image using LibTIFF library
    obj = Tiff(filename, 'r');
    offsets = getTag(obj, 'SubIFD');
    setSubDirectory(obj, offsets(1));
    rawim = read(obj);
    close(obj);

    % Read useful metadata
    meta_info = imfinfo(filename);
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

    % Perform linear transformation on rawim
    rawim = (rawim - blacklevel) / (whitelevel - blacklevel);
    rawim = max(0, min(rawim, 1));

    % Return only useful part of rawim
    rawim = rawim(y_origin:y_origin+height-1, x_origin:x_origin+width-1);


    

     % Crop the raw image to useful part
    rawim = rawim(y_origin:(y_origin + height - 1), x_origin:(x_origin + width - 1));

    % Apply the point transformation
    rawim = (rawim - blacklevel) / (whitelevel - blacklevel);

    % Clip values outside the range [0, 1]
    rawim = max(0, min(rawim, 1));
end