function[LABEL_MAP] = get_label_map_from_colors(INPUT_DATA, label_data)
% get LABEL_MAP from color input for label-maps
% INPUT_DATA is from OpenShapes App -- 
% label_data has color coding -- 

% I . get unique data points in INPUT_DATA
semantic_map = double(INPUT_DATA(:,:,1))*10 + ...
		double(INPUT_DATA(:,:,2))*100 + ...
		double(INPUT_DATA(:,:,3))*1000;

unique_points = unique(semantic_map);
LABEL_MAP = zeros(size(INPUT_DATA,1), size(INPUT_DATA,2));
for i = 1:length(unique_points)	
	if(unique_points(i) == 0)
		continue;
	end
	ith_loc_id = label_data(:,6) == unique_points(i);
	LABEL_MAP(semantic_map == unique_points(i)) = label_data(ith_loc_id,1);
end

end
