function Ccam = bilinear_demosaic(Cwb, bayertype)
    [M, N] = size(Cwb);
    Ccam = zeros(M, N, 3);
    % Creation of subdivisions based on bayertype
    switch bayertype
        case 'BGGR'
            B = Cwb(1:2:end, 1:2:end);
            G1 = Cwb(1:2:end, 2:2:end);
            G2 = Cwb(2:2:end, 1:2:end);
            R = Cwb(2:2:end, 2:2:end);
        case 'GBRG'
            G1 = Cwb(1:2:end, 1:2:end);
            B = Cwb(1:2:end, 2:2:end);
            R = Cwb(2:2:end, 1:2:end);
            G2 = Cwb(2:2:end, 2:2:end);
        case 'GRBG'
            G1 = Cwb(1:2:end, 1:2:end);
            R = Cwb(1:2:end, 2:2:end);
            B = Cwb(2:2:end, 1:2:end);
            G2 = Cwb(2:2:end, 2:2:end);
        case 'RGGB'
            R = Cwb(1:2:end, 1:2:end);
            G1 = Cwb(1:2:end, 2:2:end);
            G2 = Cwb(2:2:end, 1:2:end);
            B = Cwb(2:2:end, 2:2:end);
        otherwise
            error('Unsupported bayer pattern');
    end

    % Apply bilinear interpolation method
    Ccam(1:2:end, 1:2:end, 1) = R;
    Ccam(1:2:end, 2:2:end, 1) = conv2(R, [1, 1] / 2, 'same');
    Ccam(2:2:end, 1:2:end, 1) = conv2(R, [1; 1] / 2, 'same');
    Ccam(2:2:end, 2:2:end, 1) = conv2(R, [1, 1; 1, 1] / 4, 'same');

    Ccam(1:2:end, 1:2:end, 2) = conv2(G1, [1, 1] / 2, 'same');
    Ccam(1:2:end, 2:2:end, 2) = G1;
    Ccam(2:2:end, 1:2:end, 2) = G2;
    Ccam(2:2:end, 2:2:end, 2) = conv2(G2, [1; 1] / 2, 'same');

    
    Ccam(1:2:end, 1:2:end, 3) = conv2(B, [1, 1; 1, 1] / 4, 'same');
    Ccam(1:2:end, 2:2:end, 3) = conv2(B, [1; 1] / 2, 'same');
    Ccam(2:2:end, 1:2:end, 3) = conv2(B, [1, 1] / 2, 'same');
    Ccam(2:2:end, 2:2:end, 3) = B;
end