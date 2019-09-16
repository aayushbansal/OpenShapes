% cache the resized exemplars on disk
% so we do not have to run the re-size the 
% operation every time -- 
clc; clear all;

CACHE_XMPLAR_PATH = ['./cachedir/coco/exemplars_50x50/train2017/'];
if(~isdir(CACHE_XMPLAR_PATH))
	mkdir(CACHE_XMPLAR_PATH);
end

% --
COCO_IMAGE_PATH = ['/mnt/pcie1/user/aayushb/CoCo/dataset/coco/images/'];
COCO_LABEL_PATH = ['/mnt/pcie1/user/aayushb/CoCo/dataset/coco/annotations/semantic_'];
COCO_INST_PATH = ['/mnt/pcie1/user/aayushb/CoCo/dataset/coco/annotations/panoptic_'];
XMPLARS_LIST = dir([COCO_LABEL_PATH, 'train2017/*.png']);
XMPLARS_LIST = {XMPLARS_LIST.name};
XMPLAR_LABEL_PATH = [COCO_LABEL_PATH, 'train2017/'];
WIN_SIZE = 50;

for i = 1:length(XMPLARS_LIST)

	display(['Label : ', XMPLARS_LIST{i}]);
	ith_xmp_pix = imresize(imread([XMPLAR_LABEL_PATH, XMPLARS_LIST{i}]),...
                                         [WIN_SIZE, WIN_SIZE], 'nearest');
	imwrite(ith_xmp_pix, [CACHE_XMPLAR_PATH, XMPLARS_LIST{i}]);
end

