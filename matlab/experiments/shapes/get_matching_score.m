function[score] = get_matching_score(query, nn, RES, WIN_SIZE,...
				     SHAPE_THRESH, TOP_K)

% 
query_shape = query.comp_shape;
q_h = size(query_shape,1); q_w = size(query_shape,2);
query_shape_rs = imresize(query_shape, [WIN_SIZE,WIN_SIZE], 'nearest');
query_shape(query_shape==0) =  -1;
query_shape_rs(query_shape_rs==0) = -1;

query_context = query.comp_context;
query_context_rs = imresize(query_context, [WIN_SIZE,WIN_SIZE], 'nearest');
query_label = query.shape_label;
query_ar = query.ar;

% look at all the labels in nn
train_labs = [nn(:).shape_label];

% ids with same label
train_ids = find(train_labs == query_label);
if(isempty(train_ids))
	score = [];
	return;
end

synth_data = [];
for j = 1:length(train_ids)

	jth_shape = nn(train_ids(j)).comp_shape;
	% this shape should be atleast bigger than 
	% RES * size of the query -- 
	jth_h = size(jth_shape,1); jth_w = size(jth_shape,2);
	if((jth_h < (RES*q_h)) | (jth_w < (RES*q_w)))
		continue;	
	end	

	% the aspect ratio should not be too skewed -- 
	% (ar1/ar2) > 0.5 and (ar1/ar2)< 2
	jth_ar = nn(train_ids(j)).ar;
	ar12 = query_ar/jth_ar;	
	if((ar12 < 0.5) | (ar12 > 2.0))
		continue;
	end

	% deform the shape to [100,100]
	jth_search_shape = imresize(jth_shape, [WIN_SIZE,WIN_SIZE], 'nearest');
	jth_search_shape(jth_search_shape==0) = -1;

	jth_score = (query_shape_rs(:)'*jth_search_shape(:))/...
                    (size(query_shape_rs,1)*size(query_shape_rs,2));

	%if(jth_score < SHAPE_THRESH)
	%	continue;
	%end

	% get the context map and get the score
	%jth_context = nn(train_ids(j)).comp_context;
	%jth_context_rs = imresize(jth_context, [WIN_SIZE,WIN_SIZE], 'nearest');
	%jth_context_score = (sum(query_context_rs(:) == jth_context_rs(:)))/...
	%		    (size(query_context_rs,1)*size(query_context_rs,2)) - ...
	%		    (sum(query_context_rs(:) ~= jth_context_rs(:)))/...
        %                    (size(query_context_rs,1)*size(query_context_rs,2));

	%jth_synth_data = [1, 1, size(query_shape_rs,2), size(query_shape_rs,1),...
        %                       jth_score, jth_context_score, train_ids(j), 1];
        jth_synth_data = [1, 1, size(query_shape_rs,2), size(query_shape_rs,1),...
                               jth_score, 0, train_ids(j), 1];
	synth_data = [synth_data; jth_synth_data];
end

if(isempty(synth_data))
	score = synth_data;
	return;
end

% 
val_xmpls  = synth_data(:,5) >= SHAPE_THRESH;
if(sum(val_xmpls)==0)
	% find the best possible example
	[Ys,Is] = max(synth_data(:,5));
	score = repmat(synth_data(Is,:), TOP_K,1);
	return;
end	

score = synth_data(val_xmpls,:);
if(sum(val_xmpls) < TOP_K)
	[Ys, Is] = sort(score(:,5), 'descend');
	score = score(Is,:);
	score = repmat(score, TOP_K,1);
	score = score(1:TOP_K,:);
end

end

