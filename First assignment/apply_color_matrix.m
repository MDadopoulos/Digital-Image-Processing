
%This function applies a color_matrix to a 3 channel image Cbef and returns Cafter. 
function Cafter = apply_color_matrix(Cbef, color_matrix)
    [M, N, ~] = size(Cbef);
    Cafter = zeros(M, N, 3);
    for i = 1:M
        for j = 1:N
            Cafter(i, j, :) = reshape(color_matrix * reshape(Cbef(i, j, :), [3, 1]), [1, 1, 3]);
        end
    end
end
