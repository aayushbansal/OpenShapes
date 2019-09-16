% get the list of components 
function[comp_list] = get_exemplar_shape_components(...
			xmplar_list, LABEL_PATH, INST_PATH,...
			IMAGE_PATH, label_data,  IGNORE_LABELS)

% read the segmentation file -- 
iter = 1;
for nm = 1:length(xmplar_list)

	LABEL_MAP = imread([LABEL_PATH,  xmplar_list{nm}]);
	inst_label = imread([INST_PATH, xmplar_list{nm}]);
	img = imread([IMAGE_PATH, strrep(xmplar_list{nm}, '.png', '.jpg')]);
	if(size(img,3) == 1)
        	img = cat(3, img, img, img);
	end

	%instances = rgb2gray(inst_label);
        instances = double(inst_label(:,:,1))*100 + ...
                    double(inst_label(:,:,2))*1000 + ....
                    double(inst_label(:,:,3))*10000;

	category_list = unique(LABEL_MAP);
	category_list(category_list == IGNORE_LABELS) = [];

	% --
	for i = 1:length(category_list)

	        % for each semantic label map --
        	ith_label_map = double(LABEL_MAP);
	        ith_label_map(ith_label_map ~= category_list(i)) = -1;
        	ith_label_map(ith_label_map == category_list(i)) = 1;
	        ith_label_map(ith_label_map == -1) = 0;

       		% check if this category belong to things or stuff -- 
        	is_thing = get_things_or_stuff(category_list(i), label_data);

        	% take the full component if it is not thing -- 
        	if(~is_thing)
                	[i_r, i_c] = find(ith_label_map==1);
                 	y1 = min(i_r); y2 = max(i_r);
                 	x1 = min(i_c); x2 = max(i_c);
                 	comp_shape = ith_label_map(y1:y2,x1:x2);
                 	comp_context = double(LABEL_MAP(y1:y2,x1:x2));
                 	ar = size(comp_shape,1)/size(comp_shape,2);

                        comp_rgb = double(img(y1:y2, x1:x2, :));
                        comp_rgb(:,:,1) = comp_rgb(:,:,1).*comp_shape;
                        comp_rgb(:,:,2) = comp_rgb(:,:,2).*comp_shape;
                        comp_rgb(:,:,3) = comp_rgb(:,:,3).*comp_shape;

                	comp_list(iter).comp_shape = comp_shape;
                	comp_list(iter).comp_context = comp_context;
			comp_list(iter).comp_rgb = comp_rgb;
			comp_list(iter).org_rgb = img(y1:y2, x1:x2, :);
                	comp_list(iter).shape_label = category_list(i);
                	comp_list(iter).ar = ar;
                	comp_list(iter).bbox = [y1, x1, y2, x2];
                	comp_list(iter).dim = [size(ith_label_map,1), size(ith_label_map,2)];

                	iter = iter + 1;
                	continue;
        	end

	        % get the instances if it is thing -- 
        	ith_inst_map = double(instances);
	        ith_inst_map(ith_inst_map==0) = -1;
        	ith_inst_map(ith_label_map==0) = -1;
	        ith_inst_ids = unique(ith_inst_map);
        	ith_inst_ids(ith_inst_ids == -1) = [];

	        for j = 1:length(ith_inst_ids)
	
        	        jth_inst_map = ith_inst_map;
                	jth_inst_map(jth_inst_map ~= ith_inst_ids(j)) = -1;
	                jth_inst_map(jth_inst_map == ith_inst_ids(j)) = 1;
        	        jth_inst_map(jth_inst_map == -1) = 0;

                	% --
                	[j_r, j_c] = find(jth_inst_map==1);
                 	y1 = min(j_r); y2 = max(j_r);
                 	x1 = min(j_c); x2 = max(j_c);
                 	comp_shape = jth_inst_map(y1:y2,x1:x2);
                 	comp_context = double(LABEL_MAP(y1:y2,x1:x2));
                 	ar = size(comp_shape,1)/size(comp_shape,2);

                        comp_rgb = double(img(y1:y2, x1:x2, :));
                        comp_rgb(:,:,1) = comp_rgb(:,:,1).*comp_shape;
                        comp_rgb(:,:,2) = comp_rgb(:,:,2).*comp_shape;
                        comp_rgb(:,:,3) = comp_rgb(:,:,3).*comp_shape;

                	comp_list(iter).comp_shape = comp_shape;
                	comp_list(iter).comp_context = comp_context;
                        comp_list(iter).comp_rgb = comp_rgb;
                        comp_list(iter).org_rgb = img(y1:y2, x1:x2, :);
                	comp_list(iter).shape_label = category_list(i);
                	comp_list(iter).ar = ar;
                	comp_list(iter).bbox = [y1, x1, y2, x2];
                	comp_list(iter).dim = [size(ith_label_map,1), size(ith_label_map,2)];

                	iter = iter + 1;
        	end
	end
end
