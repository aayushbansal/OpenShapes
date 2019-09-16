function[fin_exemplar_matches] = get_exemplars(LABEL_MAP, NUM_LABELS, IGNORE_LABEL,...
			 		   XMPLARS, XMPLARS_NC, XMPLARS_LIST,... 
			  		    WIN_SIZE, XMPLAR_PATH, TOP_K,...
					    IS_FROM_CACHE, CACHE_XMPLAR_PATH)
try,

	% get indicator vector --
	indicator_vector = get_indicator_vector(LABEL_MAP, NUM_LABELS,...
						 IGNORE_LABEL);

	% get exemplar matches --
	[exemplar_matches, ~] = get_exemplar_matches(indicator_vector, XMPLARS',...
						 XMPLARS_NC', XMPLARS_LIST');
	% get scores for exemplar matches -- 
	exemplar_scores = get_exemplar_scores(LABEL_MAP, WIN_SIZE, NUM_LABELS,...
					      XMPLAR_PATH, exemplar_matches, IGNORE_LABEL,...
					      IS_FROM_CACHE, CACHE_XMPLAR_PATH); 
	[Y,I] = sort(exemplar_scores, 'descend');
	num_xmpls = min(length(exemplar_matches),TOP_K);
	fin_exemplar_matches = exemplar_matches(I(1:num_xmpls));
	%fin_exemplar_feat = exemplar_feat(:,I(1:num_xmpls));	

	% verify if all the categories are covered in 
	%fin_categories = sum(fin_exemplar_feat', 1)>0;	
	%miss_categories = (fin_categories - indicator_vector)~=0;
	%if(sum(miss_categories)>0)
	%	keyboard;
	%end

catch, 

	keyboard;
	
end


end
