function gray_image = rgb2gray(rgb_image)
    % Get the dimensions of the input RGB image
    [rows, cols, ~] = size(rgb_image);

    % Initialize the output grayscale image
    gray_image = zeros(rows, cols);

    % Define the weights for the RGB channels
    R_weight = 0.2989;
    G_weight = 0.5870;
    B_weight = 0.1140;

    % Iterate through the image, applying the NTSC conversion formula to each pixel
    for row = 1:rows
        for col = 1:cols
            gray_image(row, col) = R_weight * rgb_image(row, col, 1) + ...
                                   G_weight * rgb_image(row, col, 2) + ...
                                   B_weight * rgb_image(row, col, 3);
        end
    end


end

