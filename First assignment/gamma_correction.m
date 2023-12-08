%This function applies gamma correction to the Clinear image. 
% It uses the sRGB formula for gamma correction, which has two parts:
% a linear part for very low brightness values ​​and an exponential part for higher brightness values.

function Csrgb = gamma_correction(Clinear)
    [M, N, ~] = size(Clinear);
    Csrgb = zeros(M, N, 3);
    for i = 1:M
        for j = 1:N
            for k = 1:3
                Clinear_val = Clinear(i, j, k);
                if Clinear_val <= 0.0031308
                    Csrgb(i, j, k) = 12.92 * Clinear_val;
                else
                    Csrgb(i, j, k) = 1.055 * (Clinear_val ^ (1/2.4)) - 0.055;
                end
            end
        end
    end
    
end



