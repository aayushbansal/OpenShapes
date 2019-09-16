function[] = get_final_outputs(DATA_PATH, FILE_NAME,...
				 shape_outputs, part_outputs, TOP_K, FIN_SIZE) 


% 1. smooth the part image using conv-3
% 2. dilate shape-image
% 3. add part image to the shape-image
% 4. smooth the final image using conv-3
% 5. generate a 256x256 image -- 
for i = 1:min(length(shape_outputs),TOP_K)
	% get the part im -- 
	ith_part_im = part_outputs{i};
	% smooth the part image using conv-3
	ith_part_im = imfilter(ith_part_im, ones(5)/25);	
	% dilate shape-image
	ith_shape_im = shape_outputs{i};
	ith_shape_mask = ith_shape_im(:,:,1)==0 & ...
			 ith_shape_im(:,:,2)==0 & ...
			 ith_shape_im(:,:,3)==0 ;
	se = strel('square',9);
	ith_shape_mask = imdilate(ith_shape_mask, se);

	% add missing info from part im -- 
	ith_shape_im_x = ith_shape_im(:,:,1); ith_part_im_x = ith_part_im(:,:,1);
	ith_shape_im_x(ith_shape_mask) = ith_part_im_x(ith_shape_mask);

        ith_shape_im_y = ith_shape_im(:,:,2); ith_part_im_y = ith_part_im(:,:,2); 
        ith_shape_im_y(ith_shape_mask) = ith_part_im_y(ith_shape_mask);

        ith_shape_im_z = ith_shape_im(:,:,3); ith_part_im_z = ith_part_im(:,:,3); 
        ith_shape_im_z(ith_shape_mask) = ith_part_im_z(ith_shape_mask);

	ith_shape_im_new = cat(3, ith_shape_im_x, ith_shape_im_y, ith_shape_im_z);

	% smooth the image 
	ith_shape_im_new = imresize(imfilter(ith_shape_im_new, ones(3)/9), [FIN_SIZE, FIN_SIZE]);
	if(~isdir([DATA_PATH, '/', FILE_NAME, '/']))
		mkdir([DATA_PATH, '/', FILE_NAME, '/']);
	end	

	imwrite(ith_shape_im_new, [DATA_PATH, '/', FILE_NAME, '/', num2str(i, '%02d'), '.png']);
end

