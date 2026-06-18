"""Custom losses."""
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable

__all__ = ['MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss', 'FAscnn_ppTotalLoss', 'LovaszSoftmaxLoss']

# ==============================================================================
# Lovász-Softmax Implementation (Added for fine-tuning)
# ==============================================================================

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filter(lambda x: x == x, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class LovaszSoftmaxLoss(nn.Module):
    """
    Opakowanie na Lovász-Softmax Loss, gotowe do podpięcia do segmentacji obrazu.
    """
    def __init__(self, ignore_index=255, classes='present'):
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index
        self.classes = classes

    def forward(self, logits, target):
        # logits: [B, C, H, W]
        # target: [B, H, W]
        
        # 1. Apply softmax to get probabilities
        probas = F.softmax(logits, dim=1)
        
        # 2. Flatten predictions and targets
        probas_flat = probas.permute(0, 2, 3, 1).contiguous().view(-1, probas.size(1)) # [B*H*W, C]
        target_flat = target.view(-1) # [B*H*W]
        
        # 3. Filter out ignore_index
        valid_mask = target_flat != self.ignore_index
        probas_valid = probas_flat[valid_mask]
        target_valid = target_flat[valid_mask]
        
        if target_valid.numel() == 0:
            return logits.sum() * 0.0
            
        # 4. Calculate Lovasz Loss
        return lovasz_softmax_flat(probas_valid, target_valid, classes=self.classes)

# ==============================================================================
# Existing Code
# ==============================================================================

class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_label=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_label)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs)


class SoftmaxCrossEntropyOHEMLoss(nn.Module):
    def __init__(self, ignore_label=-1, thresh=0.7, min_kept=256, use_weight=True, **kwargs):
        super(SoftmaxCrossEntropyOHEMLoss, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            print("w/ class balance")
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)


class MixSoftmaxCrossEntropyOHEMLoss(SoftmaxCrossEntropyOHEMLoss):
    def __init__(self, aux=False, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_label=ignore_index, **kwargs)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs)
        
class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, weight=None):
        super(OHEMCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.min_kept = min_kept
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))

    def forward(self, pred, target):
        pixel_losses = F.cross_entropy(
            pred, target, weight=self.weight, ignore_index=self.ignore_index, reduction='none'
        ).view(-1)
        
        mask = target.view(-1) != self.ignore_index
        valid_losses = pixel_losses[mask]
        
        if valid_losses.numel() == 0:
            return pixel_losses.sum() * 0.0

        sort_losses, _ = valid_losses.sort(descending=True)
        keep_num = min(valid_losses.numel(), self.min_kept)
        
        if sort_losses[keep_num - 1] > self.thresh:
            keep_num = (valid_losses > self.thresh).sum()

        return sort_losses[:keep_num].mean()        

class FAscnn_ppOHEMLoss(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000):
        super(FAscnn_ppOHEMLoss, self).__init__()
        self.ignore_index = ignore_index
        self.min_kept = min_kept
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        
        # self.weight = torch.tensor([
        #     0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 
        #     0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 
        #     1.0865, 1.1529, 1.0507
        # ])
        
        self.weight = torch.tensor([
            0.8373, 0.9180, 0.8660, 
            1.5517, # (3) Wall: było 1.0345 -> x1.5
            1.5249, # (4) Fence: było 1.0166 -> x1.5
            1.4953, # (5) Pole: było 0.9969 -> x1.5
            0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 
            2.2232, # (12) Rider: było 1.1116 -> x2.0 
            0.9037, 1.0865, 1.0955, 1.0865, 
            2.3058, # (17) Motorcycle: było 1.1529 -> x2.0 
            1.0507
        ])
               
        
    def forward(self, pred, target):
        if self.weight.device != pred.device:
            self.weight = self.weight.to(pred.device)

        pixel_losses = F.cross_entropy(
            pred, target, weight=self.weight, ignore_index=self.ignore_index, reduction='none'
        ).view(-1)
        
        mask = target.view(-1) != self.ignore_index
        valid_losses = pixel_losses[mask]
        
        if valid_losses.numel() == 0:
            return pixel_losses.sum() * 0.0

        sort_losses, _ = valid_losses.sort(descending=True)
        keep_num = min(valid_losses.numel(), self.min_kept)
        
        if sort_losses[keep_num - 1] > self.thresh:
            keep_num = (valid_losses > self.thresh).sum()

        return sort_losses[:keep_num].mean()

class BiSeNetOhemCELoss(nn.Module):

    def __init__(self, thresh, ignore_lb=255):
        super(BiSeNetOhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


    
    