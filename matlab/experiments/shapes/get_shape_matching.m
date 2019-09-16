function[query_scores] = get_shape_matching(query_shapes, exemplar_shapes,...
					    RES, WIN_SIZE, SHAPE_THRESH, TOP_K)

% -- 
try,

	for i = 1:length(query_shapes)

		ith_score = get_matching_score(query_shapes(i), exemplar_shapes,...
					       RES, WIN_SIZE, SHAPE_THRESH, TOP_K);
		if(isempty(ith_score))
			query_scores(i).score = [];
		else
        	        query_scores(i).score = ith_score;
		end
	end
catch,
	keyboard;
end

end
