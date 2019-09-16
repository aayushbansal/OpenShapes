# this script is used to create indicator vector of exemplar set
import json
import numpy as np
from PIL import Image
import os

COCO_ANNOTATIONS_PATH = './dataset/' \
                        'coco/panopticapi/panoptic_coco_categories.json';

DUMP_DATA_PATH = './cachedir/coco/init/';
if not os.path.exists(DUMP_DATA_PATH):
	os.mkdir(DUMP_DATA_PATH)

# load annotations -
with open(COCO_ANNOTATIONS_PATH) as f:
	annotations = json.load(f)

label_data = np.zeros((len(annotations),4))
for i in range(0,len(annotations)):
	label_data[i,0] = annotations[i]["id"]
	label_data[i,1:4] = annotations[i]["color"]

# IMAGE PATHS --
COCO_IMAGE_PATH = './dataset/coco/images/';
COCO_LABEL_PATH = './dataset/coco/annotations/semantic_';
COCO_INST_PATH = './dataset/coco/annotations/panoptic_';

# get the list of images in training/exemplar set --
#train_imagelist = sorted(glob.glob(COCO_LABEL_PATH + 'train2017/*.png'))
train_imagelist = sorted(os.listdir(COCO_LABEL_PATH + 'train2017/'))

NUM_LABELS = 200;
train_labels = np.zeros((len(train_imagelist),NUM_LABELS),dtype=bool)
num_categories = np.zeros((len(train_imagelist),1))
for i in range(0,len(train_imagelist)):
	print("READING TRAINING DATA: " + train_imagelist[i])
	ith_train_seg = np.unique(\
			Image.open(COCO_LABEL_PATH + 'train2017/' + train_imagelist[i]))
	ith_train_seg = ith_train_seg[ith_train_seg != 0]
	train_labels[i,ith_train_seg-1] = 1
	num_categories[i] = len(ith_train_seg)

# save data to disk
np.save(DUMP_DATA_PATH + 'exemplar_indicator_vector', train_labels)
np.save(DUMP_DATA_PATH + 'exemplar_categories',num_categories)
np.save(DUMP_DATA_PATH + 'exemplar_imagelist', train_imagelist)
