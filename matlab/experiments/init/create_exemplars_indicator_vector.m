% this script is used to create indicator vector of exemplar set

clc; clear all;

JSON_PATH = ['./dataset/coco/jsonlab/'];
addpath(JSON_PATH);

COCO_ANNOTATIONS_PATH = ['./dataset/',...
                        'coco/panopticapi/panoptic_coco_categories.json'];

DUMP_DATA_PATH = ['./cachedir/coco/init/'];
if(~isdir(DUMP_DATA_PATH))
        mkdir(DUMP_DATA_PATH);
end

% load_annotations
annotations = loadjson(COCO_ANNOTATIONS_PATH);
label_data = zeros(length(annotations),4);
for i = 1:length(annotations)
        label_data(i,1) = annotations{i}.id;
        label_data(i,2:4) = annotations{i}.color;
end


COCO_IMAGE_PATH = ['./dataset/coco/images/'];
COCO_LABEL_PATH = ['./dataset/coco/annotations/semantic_'];
COCO_INST_PATH = ['./dataset/coco/annotations/panoptic_'];

train_imagelist = dir([COCO_LABEL_PATH, 'train2017/*.png']);
train_imagelist = {train_imagelist.name};

% keep the semantic label map in memory -- 
NUM_LABELS = 200;
train_labels = false(length(train_imagelist), NUM_LABELS);
num_categories = zeros(length(train_imagelist),1);

for i = 1:length(train_imagelist)

	display(['READING TRAINING DATA: ', train_imagelist{i}]);
	ith_train_seg = unique(imread([COCO_LABEL_PATH,...
                         'train2017/', train_imagelist{i}]));
	ith_train_seg(ith_train_seg==0) = [];	
	train_labels(i,ith_train_seg) = true;
	
	num_categories(i) = length(ith_train_seg);
	%train_labels = unique([train_labels; unique(ith_train_seg)]);
        %train_labels(i).labels = unique(ith_train_seg);
end

save([DUMP_DATA_PATH, 'exemplars_indicator_vector.mat'], 'train_labels', 'num_categories');
