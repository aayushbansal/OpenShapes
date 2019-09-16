import json
import numpy as np
import cv2
from experiments.patches import extract_patches
import time
import sys


class Shapes():
    def name(self):
        return 'shapes'

    def __init__(self, opt):
        self.WIN_SIZE = opt.WIN_SIZE
        self.IGNORE_LABEL = opt.IGNORE_LABEL
        self.LABEL_PATH = opt.EXEMPLAR_LABEL_PATH
        self.INST_PATH = opt.EXEMPLAR_INST_PATH
        self.IMAGE_PATH = opt.EXEMPLAR_IMAGE_PATH
        self.TOP_K = opt.TOP_K
        self.RES_F = opt.RES_F
        self.SHAPE_THRESH = opt.SHAPE_THRESH
        self.IS_PARTS = opt.IS_PARTS
        self.PARTS_WIN = opt.PARTS_WIN
        self.PARTS_SIZE = opt.PARTS_SIZE
        self.FIN_SIZE = opt.FIN_SIZE
        self.PAT_HEIGHT = opt.PAT_HEIGHT
        self.PAT_WIDTH = opt.PAT_WIDTH
        self.PAT_NBD = opt.PAT_NBD
        self.PAT_LOC = self.get_pat_loc()

        # load annotations --
        with open(opt.COCO_ANNOTATIONS_PATH) as f:
            annotations = json.load(f)
        label_data = np.zeros((len(annotations), 6))
        for i in range(0, len(annotations)):
            label_data[i, 0] = annotations[i]["id"]
            label_data[i, 1:4] = annotations[i]["color"]
            label_data[i, 4] = annotations[i]["isthing"]
            label_data[i, 5] = label_data[i, 1] + \
                label_data[i, 2]*256 + \
                label_data[i, 3]*65536
        self.label_data = label_data

        self.label_dict = {}
        for annotation in annotations:
            self.label_dict[tuple(annotation["color"])
                            ] = np.array([annotation["id"]])

    def convert_colors_to_labels_old(self, INPUT_MAP):

        semantic_map = np.int32(INPUT_MAP[:, :, 0]) + \
            np.int32(INPUT_MAP[:, :, 1])*256 + \
            np.int32(INPUT_MAP[:, :, 2])*65536
        label_data = self.label_data

        #print("Shape", INPUT_MAP.shape)
        #print np.unique(INPUT_MAP, axis=1)

        unique_points = np.unique(semantic_map)
        #print unique_points
        LABEL_MAP = np.zeros((np.size(INPUT_MAP, 0), np.size(INPUT_MAP, 1)))
        for i in range(0, len(unique_points)):
            if(unique_points[i] == self.IGNORE_LABEL):
                continue
            ith_loc_id = label_data[:, 5] == unique_points[i]
            #print ith_loc_id

            LABEL_MAP[semantic_map == unique_points[i]
                      ] = label_data[ith_loc_id, 0]
        return LABEL_MAP

    def convert_colors_to_labels(self, INPUT_MAP):

        # Get all unique pixels in input
        unique_pixels = np.unique(INPUT_MAP.reshape(-1, 3), axis=0)

        # Create a blank matrix with same dimensions
        LABEL_MAP = np.zeros((INPUT_MAP.shape[:-1]))

        for upixel in unique_pixels:
            label_of_pixel = self.label_dict.get(
                tuple(upixel), np.array([self.IGNORE_LABEL]))
            # print("LOP", label_of_pixel)
            condition = np.all(INPUT_MAP == upixel, 2)
            LABEL_MAP[condition] = label_of_pixel
        return LABEL_MAP

    def get_query_shapes(self, LABEL_MAP, INST_MAP):

        instances = INST_MAP[:, :, 0]*10 + \
            INST_MAP[:, :, 1]*100 + \
            INST_MAP[:, :, 2]*1000
        category_list = np.unique(LABEL_MAP)
        category_list = category_list[category_list != self.IGNORE_LABEL]
        query_shapes = {}
        iter_ = 0
        for n in range(0, len(category_list)):
            # for each semantic label map --
            nth_label_map = np.int16(LABEL_MAP)
            nth_label_map[nth_label_map != category_list[n]] = -1
            nth_label_map[nth_label_map == category_list[n]] = 1
            nth_label_map[nth_label_map == -1] = 0
            # ---------------------------------------------------
            # check if this category belongs to things or stuff--
            is_thing = self.label_data[self.label_data[:, 0]
                                       == category_list[n], 4]
            if(is_thing != 1):
                [n_r, n_c] = np.nonzero(nth_label_map)
                y1 = np.amin(n_r)
                y2 = np.amax(n_r)
                x1 = np.amin(n_c)
                x2 = np.amax(n_c)
                comp_shape = nth_label_map[y1:y2, x1:x2]
                comp_context = LABEL_MAP[y1:y2, x1:x2]
                ar = np.size(
                    comp_context, 0)/(float(np.size(comp_context, 1)) + sys.float_info.epsilon)
                # -- get the comp-list
                query_shapes[iter_] = {}
                query_shapes[iter_]['comp_shape'] = comp_shape
                query_shapes[iter_]['comp_context'] = comp_context
                query_shapes[iter_]['shape_label'] = category_list[n]
                query_shapes[iter_]['ar'] = ar
                query_shapes[iter_]['bbox'] = [y1, x1, y2, x2]
                query_shapes[iter_]['dim'] = [
                    np.size(nth_label_map, 0), np.size(nth_label_map, 1)]
                iter_ = iter_ + 1
            else:
                nth_inst_map = np.empty_like(instances)
                np.copyto(nth_inst_map, instances)
                nth_inst_map[nth_inst_map == 0] = -1
                nth_inst_map[nth_label_map == 0] = -1
                nth_inst_ids = np.unique(nth_inst_map)
                nth_inst_ids = nth_inst_ids[nth_inst_ids != -1]
                for m in range(0, len(nth_inst_ids)):
                    mth_inst_map = np.empty_like(nth_inst_map)
                    np.copyto(mth_inst_map, nth_inst_map)
                    mth_inst_map[mth_inst_map != nth_inst_ids[m]] = -1
                    mth_inst_map[mth_inst_map == nth_inst_ids[m]] = 1
                    mth_inst_map[mth_inst_map == -1] = 0
                    # --
                    [m_r, m_c] = np.nonzero(mth_inst_map)
                    y1 = np.amin(m_r)
                    y2 = np.amax(m_r)
                    x1 = np.amin(m_c)
                    x2 = np.amax(m_c)
                    comp_shape = mth_inst_map[y1:y2, x1:x2]
                    comp_context = LABEL_MAP[y1:y2, x1:x2]
                    ar = np.size(
                        comp_context, 0)/(float(np.size(comp_context, 1)) + sys.float_info.epsilon)
                    # -- comp_list data
                    query_shapes[iter_] = {}
                    query_shapes[iter_]['comp_shape'] = comp_shape
                    query_shapes[iter_]['comp_context'] = comp_context
                    query_shapes[iter_]['shape_label'] = category_list[n]
                    query_shapes[iter_]['ar'] = ar
                    query_shapes[iter_]['bbox'] = [y1, x1, y2, x2]
                    query_shapes[iter_]['dim'] = [
                        np.size(nth_label_map, 0), np.size(nth_label_map, 1)]
                    iter_ = iter_ + 1
        return query_shapes

    def get_exemplar_shapes(self, EXEMPLAR_MATCHES):
        exemplar_shapes = {}
        iter_ = 0
        for i in range(0, len(EXEMPLAR_MATCHES)):
            LABEL_MAP = cv2.imread(self.LABEL_PATH + EXEMPLAR_MATCHES[i][0])
            LABEL_MAP = np.int16(LABEL_MAP[:, :, 0])
            INST_MAP = np.int32(cv2.imread(
                self.INST_PATH + EXEMPLAR_MATCHES[i][0]))
            EXEMPLAR_IMAGE = cv2.imread(self.IMAGE_PATH +
                                        (EXEMPLAR_MATCHES[i][0]).replace('.png', '.jpg'))
            instances = INST_MAP[:, :, 0]*10 + \
                INST_MAP[:, :, 1]*100 + \
                INST_MAP[:, :, 2]*1000
            category_list = np.unique(LABEL_MAP)
            category_list = category_list[category_list != self.IGNORE_LABEL]
            for n in range(0, len(category_list)):
                # for each semantic label map --
                nth_label_map = np.empty_like(LABEL_MAP)
                np.copyto(nth_label_map, LABEL_MAP)
                nth_label_map[nth_label_map != category_list[n]] = -1
                nth_label_map[nth_label_map == category_list[n]] = 1
                nth_label_map[nth_label_map == -1] = 0
                # ---------------------------------------------------
                # check if this category belongs to things or stuff--
                is_thing = self.label_data[self.label_data[:, 0]
                                           == category_list[n], 4]
                if(is_thing != 1):
                    [n_r, n_c] = np.nonzero(nth_label_map)
                    y1 = np.amin(n_r)
                    y2 = np.amax(n_r)
                    x1 = np.amin(n_c)
                    x2 = np.amax(n_c)
                    comp_shape = nth_label_map[y1:y2, x1:x2]
                    comp_context = LABEL_MAP[y1:y2, x1:x2]

                    comp_rgb = np.empty_like(EXEMPLAR_IMAGE[y1:y2, x1:x2, :])
                    np.copyto(comp_rgb, EXEMPLAR_IMAGE[y1:y2, x1:x2, :])
                    comp_rgb[:, :, 0] = comp_rgb[:, :, 0] * comp_shape
                    comp_rgb[:, :, 1] = comp_rgb[:, :, 1] * comp_shape
                    comp_rgb[:, :, 2] = comp_rgb[:, :, 2] * comp_shape
                    ar = np.size(
                        comp_context, 0)/(float(np.size(comp_context, 1)) + sys.float_info.epsilon)
                    # -- get the comp-list
                    exemplar_shapes[iter_] = {}
                    exemplar_shapes[iter_]['comp_shape'] = comp_shape
                    exemplar_shapes[iter_]['comp_context'] = comp_context
                    exemplar_shapes[iter_]['comp_rgb'] = comp_rgb
                    exemplar_shapes[iter_]['org_rgb'] = EXEMPLAR_IMAGE[y1:y2, x1:x2, :]
                    exemplar_shapes[iter_]['shape_label'] = category_list[n]
                    exemplar_shapes[iter_]['ar'] = ar
                    exemplar_shapes[iter_]['bbox'] = [y1, x1, y2, x2]
                    exemplar_shapes[iter_]['dim'] = [
                        np.size(nth_label_map, 0), np.size(nth_label_map, 1)]
                    iter_ = iter_ + 1
                else:
                    nth_inst_map = np.empty_like(instances)
                    np.copyto(nth_inst_map, instances)
                    nth_inst_map[nth_inst_map == 0] = -1
                    nth_inst_map[nth_label_map == 0] = -1
                    nth_inst_ids = np.unique(nth_inst_map)
                    nth_inst_ids = nth_inst_ids[nth_inst_ids != -1]
                    for m in range(0, len(nth_inst_ids)):
                        mth_inst_map = np.empty_like(nth_inst_map)
                        np.copyto(mth_inst_map, nth_inst_map)
                        mth_inst_map[mth_inst_map != nth_inst_ids[m]] = -1
                        mth_inst_map[mth_inst_map == nth_inst_ids[m]] = 1
                        mth_inst_map[mth_inst_map == -1] = 0
                        # --
                        [m_r, m_c] = np.nonzero(mth_inst_map)
                        y1 = np.amin(m_r)
                        y2 = np.amax(m_r)
                        x1 = np.amin(m_c)
                        x2 = np.amax(m_c)
                        comp_shape = mth_inst_map[y1:y2, x1:x2]
                        comp_context = LABEL_MAP[y1:y2, x1:x2]

                        comp_rgb = np.empty_like(
                            EXEMPLAR_IMAGE[y1:y2, x1:x2, :])
                        np.copyto(comp_rgb, EXEMPLAR_IMAGE[y1:y2, x1:x2, :])
                        comp_rgb[:, :, 0] = comp_rgb[:, :, 0] * comp_shape
                        comp_rgb[:, :, 1] = comp_rgb[:, :, 1] * comp_shape
                        comp_rgb[:, :, 2] = comp_rgb[:, :, 2] * comp_shape
                        ar = np.size(
                            comp_context, 0)/(float(np.size(comp_context, 1)) + sys.float_info.epsilon)
                        # -- comp_list data
                        exemplar_shapes[iter_] = {}
                        exemplar_shapes[iter_]['comp_shape'] = comp_shape
                        exemplar_shapes[iter_]['comp_context'] = comp_context
                        exemplar_shapes[iter_]['comp_rgb'] = comp_rgb
                        exemplar_shapes[iter_]['org_rgb'] = EXEMPLAR_IMAGE[y1:y2, x1:x2, :]
                        exemplar_shapes[iter_]['shape_label'] = category_list[n]
                        exemplar_shapes[iter_]['ar'] = ar
                        exemplar_shapes[iter_]['bbox'] = [y1, x1, y2, x2]
                        exemplar_shapes[iter_]['dim'] = [
                            np.size(nth_label_map, 0), np.size(nth_label_map, 1)]
                        iter_ = iter_ + 1

        return exemplar_shapes

    def get_shapes(self, LABEL_MAP, INST_MAP, EXEMPLAR_MATCHES):
        self.query_shapes = self.get_query_shapes(LABEL_MAP, INST_MAP)
        self.exemplar_shapes = self.get_exemplar_shapes(EXEMPLAR_MATCHES)
        return self.query_shapes, self.exemplar_shapes

    def get_matching_score(self, query_data, exemplar_data):
        # query data --
        query_shape = np.empty_like(query_data['comp_shape'])
        np.copyto(query_shape, query_data['comp_shape'])
        q_h = np.size(query_shape, 0)
        q_w = np.size(query_shape, 1)

        if(q_h <= 1 | q_w <= 1):
            return np.array([])

        query_shape_rs = cv2.resize(query_shape, dsize=(self.WIN_SIZE, self.WIN_SIZE),
                                    interpolation=cv2.INTER_NEAREST)
        query_shape_rs[query_shape_rs == 0] = -1

        query_label = query_data['shape_label']
        query_ar = query_data['ar']

        # get the relevant labels from the exemplar data
        synth_data = np.zeros((len(exemplar_data), 8), dtype='float')
        for j in range(0, len(exemplar_data)):

            jth_label = exemplar_data[j]['shape_label']
            if(jth_label != query_label):
                continue

            jth_shape = np.empty_like(exemplar_data[j]['comp_shape'])
            np.copyto(jth_shape, exemplar_data[j]['comp_shape'])

            jth_h = np.size(jth_shape, 0)
            jth_w = np.size(jth_shape, 1)
            if((jth_h < (self.RES_F * q_h)) | (jth_w < (self.RES_F * q_w))):
                continue

            jth_ar = exemplar_data[j]['ar']
            ar12 = np.divide(query_ar, float(jth_ar) + sys.float_info.epsilon)

            if((ar12 < 0.5) | (ar12 > 2.0)):
                continue

            jth_search_shape = cv2.resize(jth_shape, dsize=(self.WIN_SIZE, self.WIN_SIZE),
                                          interpolation=cv2.INTER_NEAREST)
            jth_search_shape[jth_search_shape == 0] = -1
            jth_score = np.divide((query_shape_rs.flatten() * jth_search_shape.flatten()).sum(),
                                  float(np.size(query_shape_rs, 0)*np.size(query_shape_rs, 1)) + sys.float_info.epsilon)
            synth_data[j, :] = [1, 1, np.size(query_shape_rs, 1), np.size(query_shape_rs, 0),
                                jth_score, 0, j, 1]

        synth_data = synth_data[synth_data[:, 7] == 1, :]
        if synth_data.size == 0:
            return synth_data

        # find the exmples better than SHAPE_THRESH
        val_examples = synth_data[:, 4] >= self.SHAPE_THRESH
        if(val_examples.sum() == 0):
            Is = np.argmax(synth_data[:, 4])
            score = np.tile(synth_data[Is, :], [self.TOP_K, 1])
            return score

        # if there are more examples
        score = synth_data[val_examples, :]
        Is = np.argsort(score[:, 4])
        rev_Is = Is[::-1]
        score = score[rev_Is, :]
        num_ex = np.minimum(np.size(score, 0), self.TOP_K)
        score = score[0:num_ex, :]
        if(np.size(score, 0) < self.TOP_K):
            score = np.tile(score, [self.TOP_K, 1])
            score = score[0:self.TOP_K, :]

        return score

    def get_shape_matching(self, query_shapes, exemplar_shapes):
        query_scores = {}
        for n in range(0, len(query_shapes)):
            query_scores[n] = {}
            query_scores[n]['score'] = self.get_matching_score(
                query_shapes[n], exemplar_shapes)

        return query_scores

    def get_pat_loc(self):
        CROP_HEIGHT = self.PAT_HEIGHT
        CROP_WIDTH = self.PAT_WIDTH
        PAT_NBD = self.PAT_NBD

        img_pat_loc = np.zeros(
            (CROP_HEIGHT*CROP_WIDTH, CROP_HEIGHT*CROP_WIDTH), dtype='bool')
        iter_ = 0
        for i in range(0, CROP_HEIGHT):
            for j in range(0, CROP_WIDTH):
                img_pat = np.zeros((CROP_HEIGHT, CROP_WIDTH), dtype='bool')
                img_pat[i, j] = 1

                st_pos_i = np.minimum(np.maximum(
                    i - PAT_NBD, 0), CROP_HEIGHT-1)
                st_pos_j = np.minimum(np.maximum(j - PAT_NBD, 0), CROP_WIDTH-1)
                end_pos_i = np.minimum(np.maximum(
                    st_pos_i + 2*PAT_NBD, 0), CROP_HEIGHT-1)
                end_pos_j = np.minimum(np.maximum(
                    st_pos_j + 2*PAT_NBD, 0), CROP_WIDTH-1)

                img_pat[st_pos_i:end_pos_i, st_pos_j:end_pos_j] = 1
                img_pat_loc[iter_, :] = img_pat.flatten()
                iter_ = iter_ + 1
        return img_pat_loc

    def get_part_feat(self, context_map):
        PARTS_WIN = self.PARTS_WIN
        PARTS_SIZE = self.PARTS_SIZE
        context_map_rs = cv2.resize(context_map, dsize=(
            PARTS_WIN, PARTS_WIN), interpolation=cv2.INTER_NEAREST)
        context_map_pd = -np.ones((PARTS_WIN + PARTS_SIZE*2 + 5,
                                   PARTS_WIN + PARTS_SIZE*2 + 5))
        context_map_pd[PARTS_SIZE:PARTS_SIZE+PARTS_WIN,
                       PARTS_SIZE:PARTS_SIZE+PARTS_WIN] = context_map_rs
        context_feat = extract_patches(
            context_map_pd, [PARTS_SIZE*3, PARTS_SIZE*3], 0.34)
        #part_feat = np.zeros((np.size(context_feat,1)*np.size(context_feat,2), np.size(context_feat,0)))
        # for i in range(0,np.size(context_feat,0)):
        #	part_feat[:,i] = context_feat[i,:,:].flatten()
        return context_feat

    def get_part_scores(self, query_feat, nn_feat):
        PARTS_WIN = self.PARTS_WIN
        PARTS_SIZE = self.PARTS_SIZE
        PAT_LOC = self.PAT_LOC

        part_scores = np.zeros((PARTS_WIN, 3))
        corr_map = np.zeros((PARTS_WIN, PARTS_WIN))
        for ni in range(0, PARTS_WIN):
            ni_query_feat = np.tile(query_feat[ni, :, :], [
                                    np.sum(PAT_LOC[ni, :] == 1), 1, 1])
            ni_match = ni_query_feat == nn_feat[PAT_LOC[ni, :] == 1, :, :]
            corr_map[ni, PAT_LOC[ni, :] == 1] = np.divide(np.sum(np.sum(ni_match, axis=1), axis=1),
                                                          float(np.size(ni_query_feat, 1)*np.size(ni_query_feat, 2)) + sys.float_info.epsilon)
        # for ni in range(0,PARTS_WIN):
        #	ni_query_feat = np.tile(query_feat[ni,:,:], [PARTS_WIN,1,1])
        #	ni_match = ni_query_feat == nn_feat
        #	corr_map[ni,:] = np.divide(np.sum(np.sum(ni_match, axis=1), axis=1),\
        #	 			  float(np.size(ni_query_feat,1)*np.size(ni_query_feat,2)))
            #ni_query_feat = np.tile(query_feat[:,ni], [PARTS_WIN,1]).transpose()
            #corr_map[ni,:] = np.divide(np.sum(ni_query_feat == nn_feat,axis=0), float(np.size(query_feat,0)))
        #corr_map = corr_map * PAT_LOC
        t_x = np.arange(0, PARTS_WIN - 1, PARTS_SIZE)
        t_y = np.arange(0, PARTS_WIN - 1, PARTS_SIZE)
        n_x, n_y = np.meshgrid(t_x, t_y)
        n_x = n_x.flatten()
        n_y = n_y.flatten()

        Is = np.argmax(corr_map, axis=1)
        Ys = np.max(corr_map, axis=1)
        part_scores[:, 0] = n_y[Is]
        part_scores[:, 1] = n_x[Is]
        part_scores[:, 2] = Ys
        return part_scores

    def get_part_image(self, nn_rgb, part_scores, org_mask):
        PARTS_WIN = self.PARTS_WIN
        PARTS_SIZE = self.PARTS_SIZE
        part_img = np.zeros((PARTS_WIN, PARTS_WIN, 3))
        nn_rgb_rs = cv2.resize(nn_rgb, dsize=(PARTS_WIN, PARTS_WIN))

        t_x = np.arange(0, PARTS_WIN - 1, PARTS_SIZE)
        t_y = np.arange(0, PARTS_WIN - 1, PARTS_SIZE)
        n_x, n_y = np.meshgrid(t_x, t_y)
        n_x = n_x.flatten()
        n_y = n_y.flatten()

        for i in range(0, np.size(n_x, 0)):
            ith_data = nn_rgb_rs[int(part_scores[i, 0]):int(part_scores[i, 0])+PARTS_SIZE,
                                 int(part_scores[i, 1]):int(part_scores[i, 1])+PARTS_SIZE, :]
            part_img[n_y[i]:n_y[i]+PARTS_SIZE,
                     n_x[i]:n_x[i]+PARTS_SIZE, :] = ith_data
        part_img = cv2.resize(part_img, dsize=(
            np.size(org_mask, 1), np.size(org_mask, 0)))
        part_img[:, :, 0] = part_img[:, :, 0]*org_mask
        part_img[:, :, 1] = part_img[:, :, 1]*org_mask
        part_img[:, :, 2] = part_img[:, :, 2]*org_mask
        return part_img

    def get_shape_composition(self, LABEL_MAP, query_shapes, query_scores, exemplar_shapes):

        image_outputs = {}
        for i in range(0, self.TOP_K):
            image_outputs[i] = {}
            image_outputs[i]['shape_im'] = np.zeros(
                (np.size(LABEL_MAP, 0), np.size(LABEL_MAP, 1), 3))
            image_outputs[i]['part_im'] = np.zeros(
                (np.size(LABEL_MAP, 0), np.size(LABEL_MAP, 1), 3))
            image_outputs[i]['comp_mask'] = np.zeros(
                (np.size(LABEL_MAP, 0), np.size(LABEL_MAP, 1)))
            image_outputs[i]['mask'] = np.zeros(
                (np.size(LABEL_MAP, 0), np.size(LABEL_MAP, 1)))

        for i in range(0, len(query_shapes)):
            ith_score = query_scores[i]['score']
            if ith_score.size == 0:
                continue

            # get the box info--
            ith_bbx = query_shapes[i]['bbox']
            ith_org_mask = query_shapes[i]['comp_shape']

            # get the parts feat --
            if self.IS_PARTS == 1:
                ith_part_feat = self.get_part_feat(
                    query_shapes[i]['comp_context'])

            for l in range(0, np.size(ith_score, 0)):
                lth_nn_img = np.zeros(
                    (np.size(LABEL_MAP, 0), np.size(LABEL_MAP, 1), 3))
                lth_nn_part_img = np.zeros(
                    (np.size(LABEL_MAP, 0), np.size(LABEL_MAP, 1), 3))
                lth_nn = exemplar_shapes[ith_score[l, 6]]

                lth_nn_rgb = np.empty_like(lth_nn['comp_rgb'])
                np.copyto(lth_nn_rgb, lth_nn['comp_rgb'])

                lth_nn_context = np.empty_like(lth_nn['comp_context'])
                np.copyto(lth_nn_context, lth_nn['comp_context'])

                lth_nn_rgb = cv2.resize(lth_nn_rgb, dsize=(
                    np.size(ith_org_mask, 1), np.size(ith_org_mask, 0)))
                lth_nn_rgb[:, :, 0] = lth_nn_rgb[:, :, 0]*ith_org_mask
                lth_nn_rgb[:, :, 1] = lth_nn_rgb[:, :, 1]*ith_org_mask
                lth_nn_rgb[:, :, 2] = lth_nn_rgb[:, :, 2]*ith_org_mask
                lth_nn_img[ith_bbx[0]:ith_bbx[2],
                           ith_bbx[1]:ith_bbx[3], :] = lth_nn_rgb

                if self.IS_PARTS == 1:
                    lth_part_feat = self.get_part_feat(lth_nn_context)
                    lth_part_scores = self.get_part_scores(
                        ith_part_feat, lth_part_feat)
                    lth_nn_part_img[ith_bbx[0]:ith_bbx[2], ith_bbx[1]:ith_bbx[3], :] = \
                        self.get_part_image(lth_nn['org_rgb'], lth_part_scores,
                                            ith_org_mask)
                    image_outputs[l]['part_im'] = image_outputs[l]['part_im'] + \
                        lth_nn_part_img

                image_outputs[l]['shape_im'] = image_outputs[l]['shape_im'] + lth_nn_img
                image_outputs[l]['comp_mask'] = ~((lth_nn_img[:, :, 0] == 0) & (
                    lth_nn_img[:, :, 1] == 0) & (lth_nn_img[:, :, 2] == 0))
                image_outputs[l]['mask'] = image_outputs[l]['mask'] + \
                    image_outputs[l]['comp_mask']

        return image_outputs

    def get_outputs(self, LABEL_MAP, INST_MAP, EXEMPLAR_MATCHES):
        # get query shapes and exemplar shapes
        #query_shapes = Components()
        #exemplar_shapes = Components()
        [query_shapes, exemplar_shapes] = self.get_shapes(
            LABEL_MAP, INST_MAP, EXEMPLAR_MATCHES)
        # do shape matching
        query_scores = self.get_shape_matching(query_shapes, exemplar_shapes)
        # do composition to make image
        image_outputs = self.get_shape_composition(
            LABEL_MAP, query_shapes, query_scores, exemplar_shapes)

        return image_outputs

    def finalize_images(self, image_outputs):
        fin_outputs = {}
        TOP_K = self.TOP_K
        for i in range(0, np.minimum(len(image_outputs), TOP_K)):
            # get the part im --
            ith_part_im = np.empty_like(image_outputs[i]['part_im'])
            np.copyto(ith_part_im, image_outputs[i]['part_im'])
            # smooth the part im using conv-3 filfer
            #smooth_ = np.ones((5,5),np.float32)/25
            #ith_part_im = cv2.filter2D(ith_part_im,-1,smooth_)
            ith_part_im = cv2.blur(ith_part_im, (5, 5))

            # get shape -im
            ith_shape_im = np.empty_like(image_outputs[i]['shape_im'])
            np.copyto(ith_shape_im, image_outputs[i]['shape_im'])

            ith_comp_mask = image_outputs[i]['mask']
            ith_shape_im[:, :, 0] = np.divide(
                ith_shape_im[:, :, 0], ith_comp_mask + sys.float_info.epsilon)
            ith_shape_im[:, :, 1] = np.divide(
                ith_shape_im[:, :, 1], ith_comp_mask + sys.float_info.epsilon)
            ith_shape_im[:, :, 2] = np.divide(
                ith_shape_im[:, :, 2], ith_comp_mask + sys.float_info.epsilon)

            # get shape_mask
            ith_shape_mask = ith_comp_mask == 0

            # dilate shape mask
            se = np.ones((9, 9), dtype='uint8')
            ith_dil_shape_mask = cv2.dilate(
                np.uint8(ith_shape_mask), se, iterations=1)

            # add missing for from part im --
            ith_shape_im_x = ith_shape_im[:, :, 0]
            ith_part_im_x = ith_part_im[:, :, 0]
            ith_shape_im_x[ith_dil_shape_mask ==
                           1] = ith_part_im_x[ith_dil_shape_mask == 1]

            ith_shape_im_y = ith_shape_im[:, :, 1]
            ith_part_im_y = ith_part_im[:, :, 1]
            ith_shape_im_y[ith_dil_shape_mask ==
                           1] = ith_part_im_y[ith_dil_shape_mask == 1]

            ith_shape_im_z = ith_shape_im[:, :, 2]
            ith_part_im_z = ith_part_im[:, :, 2]
            ith_shape_im_z[ith_dil_shape_mask ==
                           1] = ith_part_im_z[ith_dil_shape_mask == 1]

            ith_shape_part_im = np.empty_like(ith_shape_im)
            ith_shape_part_im[:, :, 0] = ith_shape_im_x
            ith_shape_part_im[:, :, 1] = ith_shape_im_y
            ith_shape_part_im[:, :, 2] = ith_shape_im_z

            # --
            fin_outputs[i] = {}
            fin_outputs[i]['im'] = cv2.resize(cv2.blur(ith_shape_part_im, (3, 3)),
                                              dsize=(self.FIN_SIZE, self.FIN_SIZE))

        return fin_outputs
