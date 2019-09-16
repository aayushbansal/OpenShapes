% make video for each of COCO val set -- 
clc; clear all;

% -- 
VID_GEN_PATH = './cachedir/coco/shapes_parts_vid_vis/';
if(~isdir(VID_GEN_PATH))
	mkdir(VID_GEN_PATH);
end
DATA_PATH = './cachedir/coco/shapes_parts/';
COCO_IMAGE_PATH = ['/mnt/pcie1/user/aayushb/CoCo/dataset/coco/images/'];
COCO_LABEL_PATH = ['/mnt/pcie1/user/aayushb/CoCo/dataset/coco/annotations/semantic_'];
COCO_INST_PATH = ['/mnt/pcie1/user/aayushb/CoCo/dataset/coco/annotations/panoptic_'];

val_imagelist = dir([COCO_LABEL_PATH, 'val2017/*.png']);
val_imagelist = {val_imagelist.name};
num_images = length(val_imagelist);

% get the knowledge of isthing or isstuff -- 
JSON_PATH = ['/mnt/pcie1/user/aayushb/CoCo/dataset/coco/jsonlab/'];
addpath(JSON_PATH);
COCO_ANNOTATIONS_PATH = ['/mnt/pcie1/user/aayushb/CoCo/dataset/',...
                        'coco/panopticapi/panoptic_coco_categories.json'];
annotations = loadjson(COCO_ANNOTATIONS_PATH);
label_data = zeros(length(annotations),5);
for i = 1:length(annotations)
        label_data(i,1) = annotations{i}.id;
        label_data(i,2:4) = annotations{i}.color;
        label_data(i,5) = annotations{i}.isthing;
end

WIN_SIZE = 256;

for i = 1:num_images
	
	display(['Image: ', val_imagelist{i}]);
	
	% check if file exists -- 
	if(~isdir([DATA_PATH, '/', strrep(val_imagelist{i}, '.png', '/')]))
		continue;
	end

	if(exist([VID_GEN_PATH, '/', strrep(val_imagelist{i}, '.png', '.mp4')], 'file'))
		continue;	
	end

	if(isLocked([VID_GEN_PATH, '/', strrep(val_imagelist{i}, '.png', '')]))
		continue;
	end

	mkdir([VID_GEN_PATH, '/', strrep(val_imagelist{i}, '.png', '/')]);

	% read the images in this directory -- 
	img_list = dir([DATA_PATH, '/', strrep(val_imagelist{i}, '.png', '/'), '*.png']);
	img_list = {img_list.name};

	ith_label  = imread([COCO_LABEL_PATH, 'val2017/', val_imagelist{i}]);
	ith_label_im = uint8(convert_labels_to_image(ith_label, label_data));
	ith_label_im = imresize(ith_label_im, [WIN_SIZE, WIN_SIZE], 'nearest');

	iter = 1;
	for ni = 1:length(img_list)-1
		im1 = imread([DATA_PATH, '/', strrep(val_imagelist{i}, '.png', '/'), img_list{ni}]);
		im2 = imread([DATA_PATH, '/', strrep(val_imagelist{i}, '.png', '/'), img_list{ni+1}]);
	
		for nj = 1:720
		
			nj_im = uint8(double(im1) + (double(im2) - double(im1))*nj/720);
			nj_im_lab = cat(2, ith_label_im, nj_im);
			imwrite(nj_im_lab, [VID_GEN_PATH, '/', strrep(val_imagelist{i}, '.png', '/'),...
						num2str(iter, '%06d'), '.jpg']);
			iter = iter + 1;
		end
	end
	
	% make video from this data -- 
	str_vid = ['avconv -i ', VID_GEN_PATH, '/', strrep(val_imagelist{i}, '.png', '/'), '%06d.jpg', ...
					' ', VID_GEN_PATH, '/', strrep(val_imagelist{i}, '.png', '.mp4')];
	unix(str_vid); 
	unlock([VID_GEN_PATH, '/', strrep(val_imagelist{i}, '.png', '')]);

end
