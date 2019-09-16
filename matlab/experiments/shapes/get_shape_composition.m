function[shape_outputs, part_outputs] = get_shape_composition(LABEL_MAP, query_shapes, exemplar_shapes,...
						query_scores, TOP_K, IS_PARTS, PART_WIN, PART_SIZE,...
						PAT_LOC)
	
% generate TOP-K IMAGES IN SHAPE-OUTPUTS
%% --
for i = 1:TOP_K 
	new_shape_image{i} = zeros(size(LABEL_MAP,1), size(LABEL_MAP,2),3);
	new_part_image{i} = zeros(size(LABEL_MAP,1), size(LABEL_MAP,2),3);
	new_mask{i} = zeros(size(LABEL_MAP,1), size(LABEL_MAP,2));
	new_comp_mask{i} = zeros(size(LABEL_MAP,1), size(LABEL_MAP,2));
end        

for i = 1:length(query_shapes)
	
	ith_score = query_scores(i).score;
	if(isempty(ith_score))
		continue;
	end
        
	% -- sort the scores on the basis of shape matching
	%[Ys, Is] = sort(ith_score(:,5) + ith_score(:,6), 'descend');
        [Ys, Is] = sort(ith_score(:,5), 'descend');
	num_shape_nns = min(size(ith_score,1), TOP_K);
	Ys = Ys(1:num_shape_nns); Is = Is(1:num_shape_nns);
	ith_score = ith_score(Is,:);

	% --
	ith_bbx = query_shapes(i).bbox;
	ith_org_mask = query_shapes(i).comp_shape;

	% -- get the part feature for this shape
	if(IS_PARTS)
		[ith_part_feat] = get_part_feat(query_shapes(i).comp_context, PART_WIN, PART_SIZE);
	end
	

	for l = 1:size(ith_score,1)

		% -- 
		lth_nn_img = zeros(size(LABEL_MAP,1), size(LABEL_MAP,2),3);
		lth_nn_part_img = zeros(size(LABEL_MAP,1), size(LABEL_MAP,2),3);		

		lth_nn = exemplar_shapes(ith_score(l,7));

		lth_nn_rgb = lth_nn.comp_rgb;
		lth_nn_context = lth_nn.comp_context;
		if(IS_PARTS)
			[lth_part_feat] = get_part_feat(lth_nn_context, PART_WIN, PART_SIZE);
	
			% get the score and part matching -- 
			[lth_part_scores] = get_part_scores(ith_part_feat, lth_part_feat, PART_WIN, ...
								PART_SIZE, PAT_LOC);
		end

		lth_nn_rgb = double(imresize(lth_nn_rgb, [size(ith_org_mask,1),...
	                                                  size(ith_org_mask,2)]));

		lth_nn_rgb(:,:,1) = lth_nn_rgb(:,:,1).*ith_org_mask;
		lth_nn_rgb(:,:,2) = lth_nn_rgb(:,:,2).*ith_org_mask;
		lth_nn_rgb(:,:,3) = lth_nn_rgb(:,:,3).*ith_org_mask;

		lth_nn_img(ith_bbx(1):ith_bbx(3),...
                           ith_bbx(2):ith_bbx(4),:) = lth_nn_rgb;

		if(IS_PARTS)
			lth_nn_part_img(ith_bbx(1):ith_bbx(3),...
					ith_bbx(2):ith_bbx(4),:) = ...
					    get_part_image(lth_nn.org_rgb, lth_part_scores,...
							   ith_org_mask, PART_WIN, PART_SIZE);
			new_part_image{l} = new_part_image{l} + lth_nn_part_img;	
		end
	
		new_comp_mask{l} = ~(lth_nn_img(:,:,1)==0 & lth_nn_img(:,:,2)==0 & lth_nn_img(:,:,3)==0); 
		new_shape_image{l} = new_shape_image{l} + lth_nn_img;
		new_mask{l} = new_mask{l} + new_comp_mask{l};
	end
end

for i = 1:TOP_K

	shape_outputs{i} = uint8(new_shape_image{i}./(cat(3, new_mask{i}, new_mask{i}, new_mask{i})+eps));
      	if(IS_PARTS)
   		part_outputs{i} = uint8(new_part_image{i});
	end
end

end
