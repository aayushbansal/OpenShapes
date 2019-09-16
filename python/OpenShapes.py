import os
import cv2
import numpy as np
from experiments.opt import Options
from experiments.context import Context
from experiments.shapes import Shapes
import time

opt = Options().parse()
query_list = sorted(os.listdir(opt.QUERY_LABEL_PATH))
context = Context(opt)
shapes = Shapes(opt)

for i in range(0,len(query_list)):
	# READ THE IMAGE -- 
	print('FILE NAME: ' + query_list[i])
	
	ith_DUMP_DATA_PATH = opt.DUMP_DATA_PATH + (query_list[i]).replace('.png', '/')
	if not os.path.exists(ith_DUMP_DATA_PATH):
                        os.mkdir(ith_DUMP_DATA_PATH)
	
	LABEL_MAP = cv2.imread(opt.QUERY_LABEL_PATH + query_list[i])
	LABEL_MAP = LABEL_MAP[:,:,0]
	INST_MAP = np.int32(cv2.imread(opt.QUERY_INST_PATH +  query_list[i]))

	# GET THE EXEMPLAR MATCHES -- 
	EXEMPLAR_MATCHES = context.get_exemplars(LABEL_MAP)

	# GET THE IMAGE OUTPUTS -- 
	image_outputs = shapes.get_outputs(LABEL_MAP, INST_MAP, EXEMPLAR_MATCHES)

	fin_outputs = shapes.finalize_images(image_outputs)
	
	for ni in range(0,len(image_outputs)):
		cv2.imwrite(ith_DUMP_DATA_PATH + str(ni) + '.png', fin_outputs[ni]['im'])

