function[is_thing] = get_things_or_stuff(label, label_data)

% find the label in label data and see if it is thing or stuff -- 
label_id = label_data(:,1) == label;
is_thing_data = label_data(:,5);
is_thing = is_thing_data(label_id);

end
