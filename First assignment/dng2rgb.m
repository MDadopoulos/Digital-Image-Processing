function [Csrgb , Clinear , Cxyz, Ccam] = dng2rgb(rawim , XYZ2Cam , wbcoeffs , bayertype , method , M, N)
    
    %apply the white balance to the raw image
    Cwb = apply_white_balance(double(rawim), wbcoeffs,bayertype);
    

    %step that converts Cwb with MOxNO to Cwb with MxN
    %constructs a new grid of M × N coordinates (for each color),
    % so that each of its four corner points coincides with the corresponding corner points of its grid Cwb image 
    Cwb = scale_grid(Cwb,M, N);
    
    % Get the Bayer pattern image and interpolate using the selected method
    % to take Ccam image
    switch method
        case 'nearest'
            Ccam = nearest_neighbor_demosaic(Cwb, bayertype);
        case 'linear'
            Ccam = bilinear_demosaic(Cwb, bayertype);
        otherwise
            error('Unsupported interpolation method');
    end
 

    % Convert from camera color space to XYZ color space, we use the inverse (inv) matrix of XYZ2Cam.
    Cxyz = apply_color_matrix(Ccam, inv(XYZ2Cam)./ repmat(sum(inv(XYZ2Cam),2),1,3)); 
    
    %keep image clipped b/w 0-1
    Cxyz = max(0,min(Cxyz,1));

    XYZ2sRGB=[3.2406 -1.5372 -0.4986; -0.9689 1.8758 0.0415;0.0557 -0.2040 1.0570];
    % Convert from XYZ color space to sRGB color space (linear)
    Clinear = apply_color_matrix(Cxyz, XYZ2sRGB);
    
    %keep image clipped b/w 0-1
    Clinear = max(0,min(Clinear,1));

    %Apply brightness correction to Rgb linear image

    %find gray image
    gray_image=rgb2gray(Clinear);
    grayscale = 0.25/mean(gray_image(:));
    bright_srgb =  min(1,Clinear*grayscale);
    
    %brightness correction for each channel
    %scale1 = 0.25/mean(Clinear(:,:,1),'all');
    %scale2 = 0.25/mean(Clinear(:,:,2),'all');
    %scale3 = 0.25/mean(Clinear(:,:,3),'all');
    %Brightness Correction
    %bright_srgb(:,:,1) =  min(1,Clinear(:,:,1)*scale1);
    %bright_srgb(:,:,2) =  min(1,Clinear(:,:,2)*scale2);
    %bright_srgb(:,:,3) =  min(1,Clinear(:,:,3)*scale3);

    %Apply gamma correction to sRGB linear image
    Csrgb = gamma_correction(bright_srgb);
    
    %keep image clipped b/w 0-1
    Csrgb = max(0,min(Csrgb,1));
end



