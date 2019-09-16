function[part_scores] = get_part_scores(feat_1, feat_2, WIN_SIZE,...
					 PART_SIZE, PAT_LOC)

try, 
	% for the query shape
	part_scores = zeros(WIN_SIZE, 3);
	corr_map = zeros(WIN_SIZE, WIN_SIZE);
	%for ni = 1:WIN_SIZE
	%	ni_feat_1 = repmat(feat_1(:,ni), 1, WIN_SIZE); 
	%	ni_score = sum(ni_feat_1 == feat_2,1)/(size(ni_feat_1,1));
	%	corr_map(ni,:) = ni_score;
	%end
	%corr_map = corr_map.*PAT_LOC;
	for ni = 1:WIN_SIZE
		ni_feat_1 = repmat(feat_1(:,ni), 1, sum(PAT_LOC(ni,:)==1));
		ni_score = sum(ni_feat_1 == feat_2(:,PAT_LOC(ni,:)==1),1)/(size(ni_feat_1,1));		
		corr_map(ni, PAT_LOC(ni,:)==1) = ni_score;
	end
	[n_x, n_y] = meshgrid(1:PART_SIZE:WIN_SIZE, 1:PART_SIZE:WIN_SIZE);
	n_pix_xy = [n_y(:), n_x(:)];
	
	[Ys, Is] = max(corr_map, [], 2);
	part_scores(:,1) = n_y(Is);
	part_scores(:,2) = n_x(Is);
	part_scores(:,3) = Ys;
catch, 
	keyboard;
end

end

