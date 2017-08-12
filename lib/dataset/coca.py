import cv2
import os
import numpy as np

from imdb import IMDB
from bbox import cal_rotated_bbox
from my_eval import voc_eval

class Coca(IMDB):
    def __init__(self, image_set, root_path, devkit_path, result_path=None):
        """
        fill basic information to initialize imdb
        :param image_set: 2007_trainval, 2007_test, etc
        :param root_path: 'selective_search_data' and 'cache'
        :param devkit_path: data and results
        :return: imdb object
        """
        super(Coca, self).__init__('coca_', image_set, root_path, devkit_path, result_path)  # set self.name

        self.root_path = root_path
        self.devkit_path = devkit_path
        self.data_path = devkit_path

        self.class_mapping = {}
        class_mapping_path = os.path.join(self.data_path, 'class_mapping.txt')
        if os.path.exists(class_mapping_path):
            with open(class_mapping_path) as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            for line in lines:
                parts = line.strip().split()
                self.class_mapping[parts[0]] = parts[1]

        classes = ['__background__']
        with open(os.path.join(self.data_path, 'classes.txt')) as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        classes += lines
        mapped_classes = []
        for cls in classes:
            mapped_class = self.get_mapped_class(cls)
            if mapped_class not in mapped_classes:
                mapped_classes.append(mapped_class)
        self.classes = tuple(mapped_classes)
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images
        # self.mask_size = mask_size
        # self.binary_thresh = binary_thresh

        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

    def get_mapped_class(self, cls):
        if cls not in self.class_mapping.keys():
            return cls
        return self.class_mapping[cls]

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, 'images', index + '.jpg')
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        return [self.load_annotation(index) for index in self.image_set_index]

    def load_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from txt file
        :param index: index of a specific image
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)

        filename = os.path.join(self.data_path, 'annotations', index + '.txt')
        height, width = cv2.imread(roi_rec['image']).shape[:2]
        roi_rec['height'] = float(height)
        roi_rec['width'] = float(width)

        with open(filename) as f:
            lines = [line for line in f.readlines() if line.strip()]
        # for i in range(len(lines) - 1, -1, -1):
        #     difficult = int(lines[i].strip().split()[5])
        #     if difficult and not self.config['use_diff']:
        #         del lines[i]
        rotate = 0
        if len(lines) > 0:
            parts = lines[0].strip().split()
            if parts[0] == 'rotate':
                del lines[0]
                rotate = int(parts[1])

        num_objs = len(lines)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs,), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        class_to_index = dict(zip(self.classes, range(self.num_classes)))

        i = 0
        for line in lines:
            split_line = line.strip().split()
            mapped_class = self.get_mapped_class(split_line[0])
            if mapped_class not in class_to_index.keys():
                continue
            cls = class_to_index[mapped_class]
            x1 = float(split_line[1])
            y1 = float(split_line[2])
            x2 = float(split_line[3])
            y2 = float(split_line[4])
            # truncated = int(split_line[5])
            # if truncated:
            #    continue
            boxes[i, :] = [x1, y1, x2, y2]
            gt_classes[i] = cls
            overlaps[i, cls] = 1.0
            i += 1
        if i < num_objs:
            boxes.resize(i, 4)
            gt_classes.resize(i)
            overlaps.resize(i, self.num_classes)
        if i == 0:
            print '{} has no target classes'.format(filename)

        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False,
                        'rotate': rotate})
        return roi_rec

    def evaluate_detections(self, detections):
        """
        top level evaluations
        :param detections: result matrix, [bbox, confidence]
        :return: None
        """
        # make all these folders for results
        result_dir = os.path.join(self.result_path, 'results')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        self.write_pascal_results(detections)
        info = self.do_python_eval()
        return info

    def get_result_file_template(self):
        """
        this is a template
        VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        :return: a string template
        """
        res_file_folder = os.path.join(self.result_path, 'results', 'VOC' + self.year, 'Main')
        comp_id = self.config['comp_id']
        filename = comp_id + '_det_' + self.image_set + '_{:s}.txt'
        path = os.path.join(res_file_folder, filename)
        return path

    def write_pascal_results(self, all_boxes):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self.get_result_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_set_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))

    def do_python_eval(self):
        """
        python evaluation wrapper
        :return: info_str
        """
        info_str = ''

        imagesetfile = os.path.join(self.data_path, self.image_set + '.txt')
        # read list of images
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = self.parse_rec(imagename)
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames))

        aps = []
        for i, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            filename = self.get_result_file_template().format(cls)
            rec, prec, ap = voc_eval(filename, imagenames, cls, recs, ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            info_str += 'AP for {} = {:.4f}\n'.format(cls, ap)
        A = np.array(aps)
        m_ap = A[~np.isnan(A)].mean()
        print('Mean AP@0.5 = {:.4f}'.format(m_ap))
        info_str += 'Mean AP@0.5 = {:.4f}\n\n'.format(m_ap)
        return info_str

    def parse_rec(self, index):
        """ Parse a PASCAL VOC xml file """
        annopath = os.path.join(self.devkit_path, 'annotations', '{:s}.txt')
        filename = annopath.format(index)
        rotate = 0
        objects = []
        with open(filename) as f:
            line = f.readline()
            while line:
                parts = line.split()
                if parts[0] == 'rotate':
                    rotate = int(parts[1])
                else:
                    obj_struct = {'name': self.get_mapped_class(parts[0])}
                    if rotate == 0:
                        obj_struct['bbox'] = [int(parts[1]),
                                              int(parts[2]),
                                              int(parts[3]),
                                              int(parts[4])]
                    else:
                        obj_struct['bbox'] = cal_rotated_bbox(int(parts[1]), int(parts[2]), int(parts[3]),
                                                              int(parts[4]), rotate, 240, 320)
                    obj_struct['truncated'] = int(parts[5])
                    obj_struct['difficult'] = 0
                    objects.append(obj_struct)
                line = f.readline()

        return objects
