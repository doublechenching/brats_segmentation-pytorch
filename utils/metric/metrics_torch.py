#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/3/28 15:38
# @Author  : Eric Ching
import numpy as np
import torch
from skimage import measure
SUPPORTED_METRICS = ['dice', 'iou', 'ap']


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """
    def __init__(self, ignore_index=None):
        """
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index

    def __call__(self, input, target):
        """
        Args:
            input: 5D probability maps torch float tensor (NxCxDxHxW)
            target: 5D ground truth torch tensor. (NxCxDxHxW)
        Return:
            intersection over union averaged over all channels
        """
        n_classes = input.size()[1]
        # batch dim must be 1
        input = input[0]
        target = target[0]
        assert input.size() == target.size()

        binary_prediction = self._binarize_predictions(input)

        if self.ignore_index is not None:
            # zero out ignore_index
            mask = target == self.ignore_index
            binary_prediction[mask] = 0
            target[mask] = 0

        # convert to uint8 just in case
        binary_prediction = binary_prediction.byte()
        target = target.byte()
        per_channel_iou = []
        for c in range(n_classes):
            per_channel_iou.append(self._jaccard_index(binary_prediction[c], target[c]))

        return torch.mean(torch.tensor(per_channel_iou))

    def _binarize_predictions(self, input):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.sum(prediction | target).float()


class AveragePrecision:
    """
    Computes Average Precision given boundary prediction and ground truth instance segmentation.
    """

    def __init__(self, threshold=0.4, iou_range=(0.5, 1.0), ignore_index=-1, min_instance_size=None,
                 use_last_target=False):
        """
        :param threshold: probability value at which the input is going to be thresholded
        :param iou_range: compute ROC curve for the the range of IoU values: range(min,max,0.05)
        :param ignore_index: label to be ignored during computation
        :param min_instance_size: minimum size of the predicted instances to be considered
        :param use_last_target: if True use the last target channel to compute AP
        """
        self.threshold = threshold
        # always have well defined ignore_index
        if ignore_index is None:
            ignore_index = -1
        self.iou_range = iou_range
        self.ignore_index = ignore_index
        self.min_instance_size = min_instance_size
        self.use_last_target = use_last_target

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW) / or 4D numpy.ndarray
        :param target: 4D or 5D ground truth instance segmentation torch long tensor / or 3D numpy.ndarray
        :return: highest average precision among channels
        """
        if isinstance(input, torch.Tensor):
            assert input.dim() == 5
            # convert to numpy array
            input = input[0].detach().cpu().numpy()  # 4D
        if isinstance(target, torch.Tensor):
            if not self.use_last_target:
                assert target.dim() == 4
                # convert to numpy array
                target = target[0].detach().cpu().numpy()  # 3D
            else:
                # if use_last_target == True the target must be 5D (NxCxDxHxW)
                assert target.dim() == 5
                target = target[0, -1].detach().cpu().numpy()  # 3D

        if isinstance(input, np.ndarray):
            assert input.ndim == 4
        if isinstance(target, np.ndarray):
            assert target.ndim == 3

        # filter small instances from the target and get ground truth label set (without 'ignore_index')
        target, target_instances = self._filter_instances(target)

        per_channel_ap = []
        n_channels = input.shape[0]
        for c in range(n_channels):
            predictions = input[c]
            # threshold probability maps
            predictions = predictions > self.threshold
            # for connected component analysis we need to treat boundary signal as background
            # assign 0-label to boundary mask
            predictions = np.logical_not(predictions).astype(np.uint8)
            # run connected components on the predicted mask; consider only 1-connectivity
            predicted = measure.label(predictions, background=0, connectivity=1)
            ap = self._calculate_average_precision(predicted, target, target_instances)
            per_channel_ap.append(ap)

        # get maximum average precision across channels
        max_ap, c_index = np.max(per_channel_ap), np.argmax(per_channel_ap)
        print(f'Max average precision: {max_ap}, channel: {c_index}')
        return max_ap

    def _calculate_average_precision(self, predicted, target, target_instances):
        recall, precision = self._roc_curve(predicted, target, target_instances)
        recall.insert(0, 0.0)  # insert 0.0 at beginning of list
        recall.append(1.0)  # insert 1.0 at end of list
        precision.insert(0, 0.0)  # insert 0.0 at beginning of list
        precision.append(0.0)  # insert 0.0 at end of list
        # make the precision(recall) piece-wise constant and monotonically decreasing
        # by iterating backwards starting from the last precision value (0.0)
        # see: https://www.jeremyjordan.me/evaluating-image-segmentation-models/ e.g.
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])
        # compute the area under precision recall curve by simple integration of piece-wise constant function
        ap = 0.0
        for i in range(1, len(recall)):
            ap += ((recall[i] - recall[i - 1]) * precision[i])
        return ap

    def _roc_curve(self, predicted, target, target_instances):
        ROC = []
        predicted, predicted_instances = self._filter_instances(predicted)

        # compute precision/recall curve points for various IoU values from a given range
        for min_iou in np.arange(self.iou_range[0], self.iou_range[1], 0.1):
            # initialize false negatives set
            false_negatives = set(target_instances)
            # initialize false positives set
            false_positives = set(predicted_instances)
            # initialize true positives set
            true_positives = set()

            for pred_label in predicted_instances:
                target_label = self._find_overlapping_target(pred_label, predicted, target, min_iou)
                if target_label is not None:
                    # update TP, FP and FN
                    if target_label == self.ignore_index:
                        # ignore if 'ignore_index' is the biggest overlapping
                        false_positives.discard(pred_label)
                    else:
                        true_positives.add(pred_label)
                        false_positives.discard(pred_label)
                        false_negatives.discard(target_label)

            tp = len(true_positives)
            fp = len(false_positives)
            fn = len(false_negatives)

            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            ROC.append((recall, precision))

        # sort points by recall
        ROC = np.array(sorted(ROC, key=lambda t: t[0]))
        # return recall and precision values
        return list(ROC[:, 0]), list(ROC[:, 1])

    def _find_overlapping_target(self, predicted_label, predicted, target, min_iou):
        """
        Return ground truth label which overlaps by at least 'min_iou' with a given input label 'p_label'
        or None if such ground truth label does not exist.
        """
        mask_predicted = predicted == predicted_label
        overlapping_labels = target[mask_predicted]
        labels, counts = np.unique(overlapping_labels, return_counts=True)
        # retrieve the biggest overlapping label
        target_label_ind = np.argmax(counts)
        target_label = labels[target_label_ind]
        # return target label if IoU greater than 'min_iou'; since we're starting from 0.5 IoU there might be
        # only one target label that fulfill this criterion
        mask_target = target == target_label
        # return target_label if IoU > min_iou
        if self._iou(mask_predicted, mask_target) > min_iou:
            return target_label
        return None

    @staticmethod
    def _iou(prediction, target):
        """
        Computes intersection over union
        """
        intersection = np.logical_and(prediction, target)
        union = np.logical_or(prediction, target)
        return np.sum(intersection) / np.sum(union)

    def _filter_instances(self, input):
        """
        Filters instances smaller than 'min_instance_size' by overriding them with 'ignore_index'
        :param input: input instance segmentation
        :return: tuple: (instance segmentation with small instances filtered, set of unique labels without the 'ignore_index')
        """
        if self.min_instance_size is not None:
            labels, counts = np.unique(input, return_counts=True)
            for label, count in zip(labels, counts):
                if count < self.min_instance_size:
                    mask = input == label
                    input[mask] = self.ignore_index

        labels = set(np.unique(input))
        labels.discard(self.ignore_index)

        return input, labels
