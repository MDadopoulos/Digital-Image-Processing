function Ccam = nearest_neighbor_demosaic(Cwb, bayertype)
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

    % Apply nearest-neighbor method
    Ccam(1:2:end, 1:2:end, 1) = R;
    Ccam(1:2:end, 2:2:end, 1) = R;
    Ccam(2:2:end, 1:2:end, 1) = R;
    Ccam(2:2:end, 2:2:end, 1) = R;

    Ccam(1:2:end, 1:2:end, 2) = G1;
    Ccam(1:2:end, 2:2:end, 2) = G1;
    Ccam(2:2:end, 1:2:end, 2) = G2;
    Ccam(2:2:end, 2:2:end, 2) = G2;

    Ccam(1:2:end, 1:2:end, 3) = B;
    Ccam(1:2:end, 2:2:end, 3) = B;
    Ccam(2:2:end, 1:2:end, 3) = B;
    Ccam(2:2:end, 2:2:end, 3) = B;
end
