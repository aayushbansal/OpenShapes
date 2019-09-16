function[query_mask_im] = convert_labels_to_image(query_mask, label_data)

% get the unique label
uniq_labels = unique(query_mask);
query_mask_im = zeros(size(query_mask,1), size(query_mask,2),3);

for i = 1:length(uniq_labels)

	ith_label = uniq_labels(i);
	ith_mask = query_mask == ith_label;
	
	% get the pos in label-data
	if(ith_label == 0)
		continue;
	else
		ith_pos = find(label_data(:,1) == ith_label);
	end
		
	% --
	tmp_ = zeros(size(query_mask,1), size(query_mask,2));
	tmp_(ith_mask) = label_data(ith_pos,2);
	query_mask_im(:,:,1) = query_mask_im(:,:,1) + tmp_;

        tmp_(ith_mask) = label_data(ith_pos,3);
        query_mask_im(:,:,2) = query_mask_im(:,:,2) + tmp_;

        tmp_(ith_mask) = label_data(ith_pos,4);
        query_mask_im(:,:,3) = query_mask_im(:,:,3) + tmp_;
	
end


end
