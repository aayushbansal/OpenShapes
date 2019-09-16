% ----------------------------------------------------------------------- %
% OpenShapes MATLAB CODE -- 
% Contact Aayush Bansal (aayushb@cs.cmu.edu) for any queries.
% This code should strictly be used for academic purposes only.
% ----------------------------------------------------------------------- %
clc; clear all;
% GET THE DATA PATH
DATA_PATH = './cachedir/coco/OpenShapes/';
COCO_IMAGE_PATH = ['./dataset/coco/images/'];
COCO_LABEL_PATH = ['./dataset/coco/annotations/semantic_'];
COCO_INST_PATH = ['./dataset/coco/annotations/panoptic_'];

val_imagelist = dir([COCO_LABEL_PATH, 'val2017/*.png']);
val_imagelist = {val_imagelist.name};
num_images = length(val_imagelist);

% get the knowledge of isthing or isstuff -- 
COCO_ANNOTATIONS_PATH = ['./dataset/coco/panopticapi/panoptic_coco_categories.json'];
annotations = loadjson(COCO_ANNOTATIONS_PATH);
label_data = zeros(length(annotations),6);
for i = 1:length(annotations)
        label_data(i,1) = annotations{i}.id;
        label_data(i,2:4) = annotations{i}.color;
	label_data(i,5) = annotations{i}.isthing;
	label_data(i,6) = double(label_data(i,2))*10 + ....
			  double(label_data(i,3))*100 + ....
			  double(label_data(i,4))*1000 ;
end

% LOAD THE EXEMPLARS IN THE MEMORY 
EXEMPLAR_PATH = ['./cachedir/coco/init/'];
load([EXEMPLAR_PATH, 'exemplars_indicator_vector.mat'],...
			 'train_labels', 'num_categories');
% -- 
NUM_LABELS = 200;
IGNORE_LABELS = [0];
XMPLARS = train_labels;
XMPLARS_NC = num_categories;
XMPLARS_LIST = dir([COCO_LABEL_PATH, 'train2017/*.png']);
XMPLARS_LIST = {XMPLARS_LIST.name};
TOP_K_GLBL = 100;
TOP_K = 5;
WIN_SIZE = 50;
XMPLAR_LABEL_PATH = [COCO_LABEL_PATH, 'train2017/'];
XMPLAR_IMAGE_PATH = [COCO_IMAGE_PATH, 'train2017/'];
XMPLAR_INST_PATH = [COCO_INST_PATH, 'train2017/'];
RES = 0.1;

IS_FROM_CACHE = true;
CACHE_XMPLAR_PATH = ['./cachedir/coco/exemplars_50x50/train2017/'];

% PARTS PARAMETERS
IS_PARTS = true;
PARTS_WIN = 256;
PARTS_SIZE = 16;
FIN_SIZE = 256;
[PAT_LOC] = get_img_pat_loc(PARTS_SIZE, PARTS_SIZE, 2);

% --
for i = 1:num_images

	% STAGE-1:
	display([num2str(i, '%04d'), '. FILE NAME: ', val_imagelist{i}]);	
	LABEL_MAP = imread([COCO_LABEL_PATH, 'val2017/', val_imagelist{i}]);
	INST_MAP = imread([COCO_INST_PATH, 'val2017/', val_imagelist{i}]);

	% SEE IF FILE EXISTS --
	if(exist([DATA_PATH, strrep(val_imagelist{i}, '.png', '/'), '01.png'], 'file'))
		continue;
	end

	if(isLocked([DATA_PATH, strrep(val_imagelist{i}, '.png', '')]))
		continue;
	end
	
	[exemplar_matches] = get_exemplars(LABEL_MAP, NUM_LABELS, IGNORE_LABELS,...
					   XMPLARS, XMPLARS_NC, XMPLARS_LIST, ...
					   WIN_SIZE, XMPLAR_LABEL_PATH, TOP_K_GLBL,...
					   IS_FROM_CACHE, CACHE_XMPLAR_PATH);

	% STAGE-2: GET SHAPES 
	% (FOR STUFFS CATEGORY, FIND THE RELEVANT SHAPES ONLY ONCE FOR MAX) --
	SHAPE_THRESH = 0.25;
	[shape_outputs, part_outputs] = get_shape_outputs(LABEL_MAP, INST_MAP, IGNORE_LABELS, label_data,...
					    XMPLAR_IMAGE_PATH, XMPLAR_LABEL_PATH, XMPLAR_INST_PATH,...
					    exemplar_matches, TOP_K, RES, 50, SHAPE_THRESH,...
					    IS_PARTS, PARTS_WIN, PARTS_SIZE, PAT_LOC);
	

	% PROCESS THE DATA AND SAVE IT ON DRIVE --
	get_final_outputs(DATA_PATH, strrep(val_imagelist{i}, '.png', ''), ...
				shape_outputs, part_outputs, TOP_K, FIN_SIZE);
	
	unlock([DATA_PATH, strrep(val_imagelist{i}, '.png', '')]);

end

