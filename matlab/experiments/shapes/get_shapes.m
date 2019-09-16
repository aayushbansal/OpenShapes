function[query_shapes, exemplar_shapes] = get_shapes(LABEL_MAP, INST_MAP, IGNORE_LABELS,...
						     label_data, xmplar_list, ...
						     LABEL_PATH, INST_PATH, IMAGE_PATH)

try,

	% get query shapes -- 
        [query_shapes] = get_query_shape_components(LABEL_MAP, INST_MAP,...
						    IGNORE_LABELS, label_data);
        if(isempty(query_shapes))
                return;
        end

        % get exemplar shapes --
	[exemplar_shapes] = get_exemplar_shape_components(xmplar_list, LABEL_PATH, ...
							INST_PATH, IMAGE_PATH, label_data, IGNORE_LABELS);

catch, 

	keyboard;
end

end
