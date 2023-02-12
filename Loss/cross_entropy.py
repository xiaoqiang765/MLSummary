"""
Author: xiao qiang
Time: 2023/2/12 20:46 
Version: env==torch py==3.9
importance:
weight: element-wise weight
class_weight: used to balance the class,解决类间样本不均衡问题
positive_weight: pos or neg weight, 如果是二分类为torch.tensor([pos_weight]), 如果是多标签分类，则shape[N,c],代表每个类别正负
样本的权重
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """
    apply element-wise weight and reduce loss
    :param loss: element-wise loss
    :param weight: element-wist weight
    :param reduction:
    :param avg_factor:
    :return:
    """
    if weight is not None:
        loss = loss*weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None, pos_weight=None):
    """
    calculate the binary CrossEntropy loss with logits
    :param pred: the prediction with shape [N, \*]
    :param label: the gt label with shape [N, \*]
    :param weight: Element-wise weight of loss with shape[N, ], defaults:None
    :param reduction: the method used to reduce the loss, optional['mean', 'sum' 'none'], defaults:mean
    :param avg_factor: average factor that is used to average the loss, defaults None
    :param class_weight: the weight for each class with shape[c]
    :param pos_weight: the positive weight for each class with shape[c], defaults none
    :return: torch.tensor
    """
    assert pred.dim() == label.dim()
    if class_weight is not None:
        N = pred.size()[0]
        class_weight = class_weight.repeat(N, 1)  # ensure that the size of class_weight is consistent with pred
    loss = F.binary_cross_entropy_with_logits(pred, label, weight=class_weight, pos_weight=pos_weight, reduction='none')
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
        loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def soft_cross_entropy(pred, label, weight=None, reduction='mean', class_weight=None, avg_factor=None):
    """
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction with shape (N, C).
            When using "mixup", the label can be float.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
    Returns:
        torch.Tensor: The calculated loss
    """
    loss = -label * F.log_softmax(pred, dim=-1)
    if class_weight is not None:
        loss *= class_weight
    loss = loss.sum(dim=-1)
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


class CrossEntropyLoss(nn.Module):
    """
    cross entropy loss summary.
    Args:
        use_sigmoid(bool): whether the prediction uses sigmoid of softmax, Defaults: False
        use_soft(bool): whether to use the soft version of CrossEntropyLoss, Defaults: False
        reduction(str): the method used to reduce the loss, options are 'none, sum, mean', defaults: mean
        loss_weight(float): weight of the loss, defaults to 1.0
        class_weight(List[float], Optional): the weight for each class with shape[c], c is the number of the classes
        pos_weight(List[float], Optional): the positive weight for each class with the shape[c], c is the number of the
            classes, only enabled in bce loss when 'use_sigmoid' is true, default:None
    """
    def __init__(self, use_sigmoid=False, use_soft=False, reduction='mean', loss_weight=1.0, class_weight=None,
                 pos_weight=None):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.use_soft = use_soft
        assert not self.use_sigmoid and self.use_soft, 'use_sigmoid and use_soft could not be set simultaneously'
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.pos_weight = pos_weight
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_soft:
            self.cls_criterion = soft_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'sum', 'mean')
        reduction = (reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        if self.pos_weight is not None and self.use_sigmoid:
            pos_weight = cls_score.new_tensor(self.pos_weight)
            kwargs.update({'pos_weight': pos_weight})
        else:
            pos_weight = None
        loss_cls = self.loss_weight*self.cls_criterion(cls_score, label, weight, class_weight=class_weight,
                                                       reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss_cls
