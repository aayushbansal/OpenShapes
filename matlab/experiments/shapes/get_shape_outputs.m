function[shape_outputs, part_outputs] = get_shape_outputs(LABEL_MAP, INST_MAP, IGNORE_LABELS, label_data,...
                                            IMAGE_PATH, LABEL_PATH, INST_PATH,...
                                            exemplar_matches, TOP_K, RES, WIN_SIZE, SHAPE_THRESH,...
					    IS_PARTS, PARTS_WIN, PARTS_SIZE, PAT_LOC)

try,
	% get the shapes for query and exemplar set
	[query_shapes, exemplar_shapes] = get_shapes(LABEL_MAP, INST_MAP, IGNORE_LABELS,...
						     label_data, exemplar_matches,...
						     LABEL_PATH, INST_PATH, IMAGE_PATH);

	% do shape matching --
	[query_scores] = get_shape_matching(query_shapes, exemplar_shapes, RES, WIN_SIZE,...
						 SHAPE_THRESH, TOP_K);	
	
	% do shape composition -- 
	[shape_outputs, part_outputs] = get_shape_composition(LABEL_MAP, query_shapes, exemplar_shapes,...
						query_scores, TOP_K, IS_PARTS, PARTS_WIN, PARTS_SIZE, PAT_LOC);

catch,

	keyboard;
end

end
