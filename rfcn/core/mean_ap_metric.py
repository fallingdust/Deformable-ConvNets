import mxnet as mx
import numpy as np
from config.config import config as cfg
from bbox.bbox_transform import bbox_pred, clip_boxes
from nms.nms import gpu_nms_wrapper


class MeanAPMetric(mx.metric.EvalMetric):
    def __init__(self, num_classes):
        self._num_classes = num_classes
        super(MeanAPMetric, self).__init__('MeanAP')
        self._use_07_metric = False
        self._max_per_image = 400
        self._thresh = 0

    def update(self, labels, preds):
        print 'test mAP: {}'.format(self._index + 1)

        i = self._index

        im_info = labels[1].asnumpy()
        assert im_info.shape[0] == 1, "Only single-image batch implemented"
        im_scale = im_info[0, 2]
        im_shape = np.round(im_info[0, :2] / im_scale)
        
        self._gt_boxes.append(labels[0].asnumpy())
        self._gt_boxes[-1][:, 0:4] = np.round(self._gt_boxes[i][:, 0:4] / im_scale)
        scores = preds[1][0].asnumpy()

        rois = preds[0].asnumpy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scale
        
        # Apply bounding-box regression deltas
        box_deltas = preds[2][0].asnumpy()
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            num_reg_classes = 2 if cfg.CLASS_AGNOSTIC else self._num_classes
            means = np.tile(np.array(cfg.TRAIN.BBOX_MEANS), num_reg_classes)
            stds = np.tile(np.array(cfg.TRAIN.BBOX_STDS), num_reg_classes)
            box_deltas *= stds
            box_deltas += means
        
        pred_boxes = bbox_pred(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_shape)

        nms = gpu_nms_wrapper(cfg.TEST.NMS, 0)

        for j in xrange(1, self._num_classes):
            inds = np.where(scores[:, j] > self._thresh)[0]
            cls_scores = scores[inds, j]
            if cfg.CLASS_AGNOSTIC:
                cls_boxes = pred_boxes[inds, 4:8]
            else:
                cls_boxes = pred_boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets)
            cls_dets = cls_dets[keep, :]
            self._all_boxes[j].append(cls_dets)

        # Limit to max_per_image detections *over all classes*
        if self._max_per_image > 0:
            image_scores = np.hstack([self._all_boxes[j][i][:, -1]
                                      for j in xrange(1, self._num_classes)])
            if len(image_scores) > self._max_per_image:
                image_thresh = np.sort(image_scores)[-self._max_per_image]
                for j in xrange(1, self._num_classes):
                    keep = np.where(self._all_boxes[j][i][:, -1] >= image_thresh)[0]
                    self._all_boxes[j][i] = self._all_boxes[j][i][keep, :]

        self._index += 1

    def reset(self):
        super(MeanAPMetric, self).reset()
        self._index = 0
        self._all_boxes = [[] for _ in xrange(self._num_classes)]
        self._gt_boxes = []

    def get(self):
        return ['mean_ap', 'recall', 'precision'], list(self.eval_mean_ap())

    def eval_mean_ap(self):
        aps = []
        tps = 0
        fps = 0
        nposs = 0
        for cls_ind in xrange(1, self._num_classes):
            ap, tp, fp, npos = self.eval_mean_ap_of_class(cls_ind, 0.5, self._use_07_metric)
            aps += [ap]
            tps += tp
            fps += fp
            nposs += npos
            recall = np.NaN if npos == 0 else tp / float(npos)
            precision = np.NaN if tp + fp == 0 else tp / float(tp + fp)
            print('AP for {} = {:.4f}, recall = {:.4f}, precision = {:.4f}'.format(cls_ind, ap, recall, precision))
        A = np.array(aps)
        m_ap = A[~np.isnan(A)].mean()
        recall = tps / float(nposs)
        precision = tps / float(tps + fps)
        print('Mean AP = {:.4f}, recall = {:.4f}, precision = {:.4f}'.format(m_ap, recall, precision))
        return m_ap, recall, precision

    def eval_mean_ap_of_class(self, cls_ind, ovthresh, use_07_metric):
        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for img_ind in xrange(self._num_images):
            R = [obj for obj in self._gt_boxes[img_ind] if obj[4] == cls_ind]
            bbox = np.array([x[:4] for x in R])
            difficult = [False] * len(R)
            det = [False] * len(R)
            npos = npos + len(R)
            class_recs[img_ind] = {'bbox': bbox,
                                   'difficult': difficult,
                                   'det': det}

        lines = []
        det_boxes = self._all_boxes[cls_ind]
        for img_ind in xrange(self._num_images):
            dets = det_boxes[img_ind]
            if dets.size == 0:
                continue
            for k in xrange(dets.shape[0]):
                # Make detection bbox 0-based as gt bbox
                lines.append('{} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}'.format(img_ind, dets[k, -1],
                               dets[k, 0], dets[k, 1],
                               dets[k, 2], dets[k, 3]))

        if npos == 0:
            return np.NAN, 0, 0, 0
        if len(lines) == 0:
            return 0, 0, 0, npos

        splitlines = [x.strip().split(' ') for x in lines]
        image_inds = [int(x[0]) for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_inds = [image_inds[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_inds)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_inds[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.voc_ap(rec, prec, use_07_metric)

        min_fp_when_tp_max = fp[tp.argmax()]
        return ap, tp.max(), min_fp_when_tp_max, npos

    @staticmethod
    def voc_ap(rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

