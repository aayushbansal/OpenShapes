% get the list of components 
function[comp_list] = get_query_shape_components(...
					LABEL_MAP, INST_MAP,...
				 	IGNORE_LABELS,label_data)

%instances = rgb2gray(INST_MAP);
instances = double(INST_MAP(:,:,1))*100 + ...
	    double(INST_MAP(:,:,2))*1000 + ....
	    double(INST_MAP(:,:,3))*10000;
category_list = unique(LABEL_MAP);
category_list(category_list == IGNORE_LABELS) = [];
iter = 1;

% 1. get the list of semantic labels
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

	 	comp_list(iter).comp_shape = comp_shape;
		comp_list(iter).comp_context = comp_context;
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

                comp_list(iter).comp_shape = comp_shape;
                comp_list(iter).comp_context = comp_context;
                comp_list(iter).shape_label = category_list(i);
                comp_list(iter).ar = ar;
                comp_list(iter).bbox = [y1, x1, y2, x2];
                comp_list(iter).dim = [size(ith_label_map,1), size(ith_label_map,2)];

                iter = iter + 1;
	end
end

end
