%This function applies the white balance to the raw image,
% multiplying each pixel by its corresponding white balance coefficient (wbcoeffs).
function Cwb = apply_white_balance(rawim, wbcoeffs,bayertype)
    m=size(rawim,1);
    n=size(rawim,2);
    mask = wbcoeffs(2)*ones(m,n); %Initialize to all green values
    switch bayertype
        case 'RGGB'
            mask(1:2:end,1:2:end) = wbcoeffs(1); %R
            mask(2:2:end,2:2:end) = wbcoeffs(3); %B
        case 'BGGR'
            mask(2:2:end,2:2:end) = wbcoeffs(1); %R
            mask(1:2:end,1:2:end) = wbcoeffs(3); %B
        case 'GRBG'
            mask(1:2:end,2:2:end) = wbcoeffs(1); %R
            mask(1:2:end,2:2:end) = wbcoeffs(3); %B
        case 'GBRG'
            mask(2:2:end,1:2:end) = wbcoeffs(1); %R
            mask(1:2:end,2:2:end) = wbcoeffs(3); %B
    end
    Cwb = rawim .* mask;


end
