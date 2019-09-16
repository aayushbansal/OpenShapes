% prune the original matches using global distribution
% and pixel coverage -- 
function[exemplar_scores] = get_exemplar_scores(LABEL_MAP, WIN_SIZE, NUM_LABELS,...
					   XMPLAR_PATH, xmplar_list, IGNORE_LABELS,...
					   IS_FROM_CACHE, CACHE_XMPLAR_PATH)

val_pix = imresize(LABEL_MAP, [WIN_SIZE, WIN_SIZE], 'nearest');
val_feat = single(val_pix(:));
val_dist = get_distribution_labels(NUM_LABELS, val_pix, IGNORE_LABELS);

exemplar_scores = zeros(length(xmplar_list),1);

for i = 1:length(xmplar_list)
	
	% read the image -- 
	if(~IS_FROM_CACHE)
		ith_xmp_pix = imresize(imread([XMPLAR_PATH, xmplar_list{i}]),...
					 [WIN_SIZE, WIN_SIZE], 'nearest');
	else
		ith_xmp_pix = imread([CACHE_XMPLAR_PATH, xmplar_list{i}]);
	end

	ith_xmp_feat = single(ith_xmp_pix(:));
	ith_xmp_dist = get_distribution_labels(NUM_LABELS, ith_xmp_feat, IGNORE_LABELS);
	
	% get the scores -- 
	dist_scr = get_glbl_nn_score(val_dist, ith_xmp_dist);
	if(isnan(dist_scr))
		dist_scr = 1;
	end	

	pix_scr = get_pix_nn_score(val_feat, ith_xmp_feat);
	if(isnan(pix_scr))
		pix_scr = 0;
	end	

	exemplar_scores(i) = 1-dist_scr+pix_scr;
end

end

function[dist_labels] = get_distribution_labels(NUM_LABELS, LABEL_MAP,...	
						IGNORE_LABELS)

        dist_labels = zeros(NUM_LABELS,1);
	category_list = unique(LABEL_MAP);
	category_list(category_list == IGNORE_LABELS) = [];	

        for i = 1:length(category_list)
		% find the corresponding location 
		% in label-data
		actv_pixls = LABEL_MAP == category_list(i);
		dist_labels(category_list(i)) = sum(actv_pixls(:));
        end

	dist_labels = single(dist_labels/(sum(dist_labels)+eps));
end


function[nn_score] = get_glbl_nn_score(val_dist, xmp_dist)

	nn_score = single(sqrt(sum((val_dist - xmp_dist).^2)));
end

function[nn_score] = get_pix_nn_score(val_feat, xmp_feat)

	nn_score = sum(val_feat == xmp_feat)/length(val_feat);
end

