function[part_img] = get_part_image(comp_rgb, part_score, mask, WIN_SIZE, PART_SIZE)

% resize the image to WIN_SIZE X WIN_SIZE
part_img = zeros(WIN_SIZE, WIN_SIZE,3);
comp_rgb_rs = imresize(comp_rgb, [WIN_SIZE, WIN_SIZE]);

[n_x, n_y] = meshgrid(1:PART_SIZE:WIN_SIZE, 1:PART_SIZE:WIN_SIZE);
n_x = n_x(:); n_y = n_y(:);

for i = 1:length(n_x)
	ith_data = comp_rgb_rs(part_score(i,1):part_score(i,1)+PART_SIZE-1,...
			       part_score(i,2):part_score(i,2)+PART_SIZE-1,:);
	part_img(n_y(i):n_y(i)+PART_SIZE-1, n_x(i):n_x(i)+PART_SIZE-1,:) = ith_data;
								
end

% mask the image -- 
part_img = imresize(part_img, [size(mask,1), size(mask,2)]);
part_img(:,:,1) = part_img(:,:,1).*double(mask);
part_img(:,:,2) = part_img(:,:,2).*double(mask);
part_img(:,:,3) = part_img(:,:,3).*double(mask);

end
