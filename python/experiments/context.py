import cv2
import numpy as np
import sys


class Context():
    def name(self):
        return 'Context'

    def __init__(self, opt):
        self.IGNORE_LABEL = opt.IGNORE_LABEL
        self.NUM_LABELS = opt.NUM_LABELS
        self.WIN_SIZE = opt.WIN_SIZE
        self.EXEMPLARS_IND = np.load(opt.EXEMPLAR_INDICATOR_VECTOR)
        self.EXEMPLARS_NC = np.load(opt.EXEMPLAR_NC)
        self.EXEMPLARS_LIST = np.load(opt.EXEMPLAR_IMAGE_LIST)
        self.TOP_K = opt.TOP_K_GLBL
        self.IS_FROM_CACHE = opt.IS_FROM_CACHE
        self.CACHE_EXEMPLAR_PATH = opt.CACHE_EXEMPLAR_PATH
        self.EXEMPLAR_PATH = opt.EXEMPLAR_LABEL_PATH

    def get_indicator_vector(self, LABEL_MAP):
        uniq_labl = np.unique(LABEL_MAP)
        uniq_labl = uniq_labl[uniq_labl != self.IGNORE_LABEL]
        indicator_vector = np.zeros((1, self.NUM_LABELS), dtype=bool)
        indicator_vector[0, uniq_labl-1] = 1
        return indicator_vector

    def get_exemplar_matches(self, indicator_vector):
        num_ = (indicator_vector*self.EXEMPLARS_IND).sum(axis=1)
        deno_ = np.concatenate(np.maximum(np.minimum(indicator_vector.sum(axis=1),
                                                     self.EXEMPLARS_NC), 1), axis=0)
        score_ = np.divide(num_, np.float32(deno_))
        exemplar_matches = self.EXEMPLARS_LIST[score_ == 1]
        return exemplar_matches

    def get_distribution_labels(self, LABEL_MAP):
        dist_labels = np.zeros((self.NUM_LABELS, 1))
        category_list = np.unique(LABEL_MAP)
        category_list = category_list[category_list != self.IGNORE_LABEL]
        for n in range(0, len(category_list)):
            actv_pixls = LABEL_MAP == category_list[n]
            dist_labels[category_list[n]-1, 0] = actv_pixls.sum()

        dist_labels = np.concatenate(dist_labels, axis=0)
        dist_labels = dist_labels/(dist_labels.sum() + sys.float_info.epsilon)
        return dist_labels

    def get_exemplar_scores(self, LABEL_MAP, EXEMPLARS_LIST):
        val_pix = cv2.resize(LABEL_MAP, dsize=(self.WIN_SIZE, self.WIN_SIZE),
                             interpolation=cv2.INTER_NEAREST)
        val_feat = np.int32(val_pix.flatten())
        val_dist = self.get_distribution_labels(val_pix)

        # exemplar_scores
        exemplar_scores = np.zeros((len(EXEMPLARS_LIST), 1))
        for n in range(0, len(EXEMPLARS_LIST)):
            # read the xmplr label map
            if(self.IS_FROM_CACHE != 1):
                XMP_MAP = cv2.imread(self.EXEMPLAR_PATH + EXEMPLARS_LIST[n])
                XMP_MAP = cv2.resize(XMP_MAP, dsize=(self.WIN_SIZE, self.WIN_SIZE),
                                     interpolation=cv2.INTER_NEAREST)
            else:
                XMP_MAP = cv2.imread(
                    self.CACHE_EXEMPLAR_PATH + EXEMPLARS_LIST[n])
            # resize xmplr label map
            XMP_MAP = XMP_MAP[:, :, 0]
            nth_xmp_feat = np.int32(XMP_MAP.flatten())
            nth_xmp_dist = self.get_distribution_labels(XMP_MAP)
            nth_glbl_nn_scr = self.get_glbl_nn_score(val_dist, nth_xmp_dist)
            nth_pix_nn_scr = self.get_pix_nn_score(val_feat, nth_xmp_feat)
            exemplar_scores[n] = 1 - nth_glbl_nn_scr + nth_pix_nn_scr

        return exemplar_scores

    def get_glbl_nn_score(self, val_dist, exmp_dist):
        glbl_nn_score = np.sqrt((np.square(val_dist - exmp_dist)).sum())
        return glbl_nn_score

    def get_pix_nn_score(self, val_feat, exmp_feat):
        pix_nn_score = np.divide(((val_feat == exmp_feat).sum()), float(
            len(val_feat) + sys.float_info.epsilon))
        return pix_nn_score

    def get_exemplars(self, LABEL_MAP):

        # get indicator vector for query element --
        indicator_vector = self.get_indicator_vector(LABEL_MAP)
        # get exemplar matches for this indicatoru vector --
        exemplar_matches = self.get_exemplar_matches(indicator_vector)
        # get exemplar scores
        exemplar_scores = self.get_exemplar_scores(LABEL_MAP, exemplar_matches)
        # sort the exexmplars on the basis of scores and return TOP_K
        I = np.argsort(exemplar_scores, axis=0)
        exemplar_matches = exemplar_matches[I]
        #I = exemplar_scores.argsort(axis=1)
        # print(exemplar_scores)
        num_examples = np.minimum(len(exemplar_matches), self.TOP_K)
        exemplars = exemplar_matches[-num_examples:]
        return exemplars
