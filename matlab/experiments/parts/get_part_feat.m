function[context_feat] = get_part_feat(context_map, WIN_SIZE, PART_SIZE)
% get the feat for part --
% resize the input mask to WIN_SIZE X WIN_SIZE
context_map_rs = imresize(context_map, [WIN_SIZE, WIN_SIZE], 'nearest');
context_map_pd = -ones(WIN_SIZE + PART_SIZE*2, WIN_SIZE + PART_SIZE*2);
context_map_pd(PART_SIZE+1:PART_SIZE+WIN_SIZE,...
	       PART_SIZE+1:PART_SIZE+WIN_SIZE) = context_map_rs;

% --
[patches, imcol, I] = im2patches(context_map_pd, [PART_SIZE*3,PART_SIZE*3], PART_SIZE);
context_feat = reshape(patches, [size(patches,1)*size(patches,2), size(patches,4)]);
context_feat(:,[PART_SIZE+1:PART_SIZE+1:end]) = [];
context_feat(:,[PART_SIZE*PART_SIZE+1:end]) = [];

end

