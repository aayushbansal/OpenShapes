function[indicator_vector] = get_indicator_vector(LABEL_MAP, NUM_LABELS,...
						  IGNORE_LABEL)

% -- 
uniq_labl = unique(LABEL_MAP);
uniq_labl(uniq_labl == IGNORE_LABEL) = [];

indicator_vector = false(1, NUM_LABELS);
indicator_vector(uniq_labl) = true;

end
