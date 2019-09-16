% given the exemplar list, get the matches for a given query
function[exemplar_matches, exemplar_feat] = get_exemplar_matches(query, XMPLARS,...
							  XMPLARS_NC, XMPLARS_LIST)

num_ = single(query)*single(XMPLARS);
deno_ = max(min(repmat(sum(query), 1, length(XMPLARS_NC)),...
	    				XMPLARS_NC),1);
score_ = num_./deno_;
exemplar_matches = XMPLARS_LIST(score_ == 1);
exemplar_feat = XMPLARS(:,score_ == 1);

end
