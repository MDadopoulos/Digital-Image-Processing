function [scaled_grid_image] = scale_grid(bayerim,M, N)

    % Create a new grid of MxN coordinates
    x = linspace(1, size(bayerim, 2), N);
    y = linspace(1, size(bayerim, 1), M);

    % Initialize the scaled Bayer image
    scaled_grid_image = zeros(M, N, size(bayerim, 3), 'like', bayerim);

    % Bilinear interpolation
    for i = 1:M
        for j = 1:N
            % Get four surrounding points
            x1 = floor(x(j));
            x2 = ceil(x(j));
            y1 = floor(y(i));
            y2 = ceil(y(i));

            % Calculate interpolation weights
            wx = x(j) - x1;
            wy = y(i) - y1;

            % Perform bilinear interpolation
            A = double(bayerim(y1, x1, :));
            B = double(bayerim(y1, x2, :));
            C = double(bayerim(y2, x1, :));
            D = double(bayerim(y2, x2, :));

            scaled_grid_image(i, j, :) = (1 - wx) * (1 - wy) * A + wx * (1 - wy) * B + (1 - wx) * wy * C + wx * wy * D;
        end
    end
end


