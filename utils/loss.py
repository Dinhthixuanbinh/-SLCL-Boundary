import torch
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn

import math
import numpy as np


def jaccard_loss(true, logits, eps=1e-7):
    """
    Computes the Jaccard loss, a.k.a the IoU loss.
    Args:
        true: the ground truth of shape [B, H, W] or [B, 1, H, W]
        logits: the output of the segmentation model (without softmax) [B, C, H, W]
        eps:

    Returns:
    The Jaccard loss
    """
    num_classes = logits.shape[1]
    true = true.to(logits.device)
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1, device=true.device)[true.squeeze(1)]
        true_1_hot = torch.moveaxis(true_1_hot, -1, 1)
        true_1_hot_f = true_1_hot[:, 0:1, :, :]  # background
        true_1_hot_s = true_1_hot[:, 1:2, :, :]  # foreground
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
        
    else:
        true_1_hot = torch.eye(num_classes, device=true.device)[true.squeeze(1)]
        true_1_hot = torch.moveaxis(true_1_hot, -1, 1)  # B, C, H, W
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true_1_hot.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return 1 - jacc_loss


def loss_calc(pred, label, gpu=0, jaccard=False):
    """
    This function returns cross entropy loss plus jaccard loss for semantic segmentation
    Args:
        pred: the logits of the prediction with shape [B, C, H, W]
        label: the ground truth with shape [B, H, W]
        gpu: the gpu number
        jaccard: if apply jaccard loss

    Returns:

    """
    label = label.long()
    if label.device != pred.device:
        label = label.to(pred.device)
    criterion = nn.CrossEntropyLoss().to(pred.device)
    loss = criterion(pred, label)
    if jaccard:
        loss = loss + jaccard_loss(true=label, logits=pred)
    return loss


def dice_loss(pred, target):
    """
    input is a torch variable of size [N,C,H,W]
    target: [N,H,W]
    """
    n, c, h, w = pred.size()
    pred = pred.to(pred.device)
    target = target.to(pred.device)
    target_onehot = torch.zeros([n, c, h, w], device=pred.device)
    target = torch.unsqueeze(target, dim=1)  # n*1*h*w
    target_onehot.scatter_(1, target.long(), 1)

    assert pred.size() == target_onehot.size(), "Input sizes must be equal."
    assert pred.dim() == 4, "Input must be a 4D Tensor."
    uniques = np.unique(target_onehot.cpu().data.numpy())
    assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

    eps = 1e-5
    probs = F.softmax(pred, dim=1)
    num = probs * target_onehot  # b,c,h,w--p*g
    num = torch.sum(num, dim=3)  # b,c,h
    num = torch.sum(num, dim=2)  # b,c,

    den1 = probs * probs  # --p^2
    den1 = torch.sum(den1, dim=3)  # b,c,h
    den1 = torch.sum(den1, dim=2)  # b,c,

    den2 = target_onehot * target_onehot  # --g^2
    den2 = torch.sum(den2, dim=3)  # b,c,h
    den2 = torch.sum(den2, dim=2)  # b,c

    dice = 2.0 * (num / (den1 + den2 + eps))  # b,c

    dice_total = torch.sum(dice) / dice.size(0)  # divide by batch_sz
    return 1 - dice_total / c


def loss_entropy(pred, device, smooth, mode='mean'):
    assert pred.ndim == 4
    assert mode == 'mean' or mode == 'sum'
    C = pred.size()[1]
    ent = pred * torch.log(pred + smooth)
    loss = (-1 / torch.log(torch.tensor(C).to(device))) * ent.sum(dim=1)
    if mode == 'mean':
        loss = loss.mean()
    elif mode == 'sum':
        loss = loss.sum(dim=tuple(np.arange(loss.ndim)[1:])).mean()
    else:
        raise NotImplementedError
    return loss


def loss_entropy_BCL(p):
    """
    entropy loss used in BCL
    :param p:
    :return:
    """
    p = F.softmax(p, dim=1)
    log_p = F.log_softmax(p, dim=1)
    loss = -torch.sum(p * log_p, dim=1)
    return loss


def cosine_similarity_BCL(class_list, label_resize, feature, label2, feature2, num_class=19):
    """
    calculate the cosine similarity between centroid of one domain and individual features of another domain
    @param class_list: the unique class index of the label
    @param label_resize: (1, 1, h, w) the label of the first domain
    @param feature: (1, C, h, w) the feature of the first domain
    @param label2: (1, 1, H, W) the label of the second domain (full size)
    @param feature2: (1, C, h, w) the feature of the second domain
    @param num_class: the total number of classes
    @return:
    """
    _, ch, feature_h, feature_w = feature.size()
    prototypes = torch.zeros(size=(num_class, ch), device=feature.device)
    for i, index in enumerate(class_list):
        if index != 255.:
            fg_mask = ((label_resize == int(index)) * 1).to(feature.device).detach()
            if fg_mask.sum() > 0:
                prototype = (fg_mask * feature).squeeze().reshape(ch, feature_h * feature_w).sum(-1) / fg_mask.sum()
            else:
                prototype = torch.zeros(ch, device=feature.device)
            prototypes[int(index)] = prototype

    cs_map = torch.matmul(F.normalize(prototypes, dim=1),
                          F.normalize(feature2.squeeze().reshape(ch, feature_h * feature_w), dim=0))
    cs_map[cs_map == 0] = -1
    cosine_similarity_map = F.interpolate(cs_map.reshape(1, num_class, feature_h, feature_w), size=label2.size()[-2:])
    cosine_similarity_map *= 10
    return cosine_similarity_map


def bidirect_contrastive_loss_BCL(feature_s, label_s, feature_t, label_t, num_class, config):
    _, ch, feature_s_h, feature_s_w = feature_s.size()
    label_s_resize = F.interpolate(label_s.float().unsqueeze(0), size=(feature_s_h, feature_s_w), mode='nearest')
    _, _, feature_t_h, feature_t_w = feature_t.size()
    label_t_resize = F.interpolate(label_t.float().unsqueeze(0), size=(feature_t_h, feature_t_w), mode='nearest')

    source_list = torch.unique(label_s_resize.float())
    target_list = torch.unique(label_t_resize.float())

    overlap_classes = [int(index.detach()) for index in source_list if index in target_list and index != 255]
    cosine_similarity_map = cosine_similarity_BCL(source_list, label_s_resize, feature_s, label_t, feature_t, num_class)

    cross_entropy_weight = torch.zeros(size=(num_class, 1), device=feature_s.device)
    cross_entropy_weight[overlap_classes] = 1
    prototype_loss = torch.nn.CrossEntropyLoss(weight=cross_entropy_weight.squeeze(), ignore_index=255)

    prediction_by_cs = F.softmax(cosine_similarity_map, dim=1)
    target_predicted = prediction_by_cs.argmax(dim=1)
    confidence_of_target_predicted = prediction_by_cs.max(dim=1).values
    masked_target_predicted = torch.where(confidence_of_target_predicted > .8, target_predicted, 255)
    masked_target_predicted_resize = F.interpolate(masked_target_predicted.float().unsqueeze(0),
                                                   size=(feature_t_h, feature_t_w), mode='nearest').long()
    label_t_resize_new = label_t_resize.clone().contiguous().long()
    label_t_resize_new[label_t_resize_new == 255] = masked_target_predicted_resize[label_t_resize_new == 255]
    if label_t_resize_new.numel() > 0:
        target_list2 = torch.unique(label_t_resize_new.float())
    else:
        target_list2 = torch.tensor([], device=feature_s.device)

    cosine_similarity_map2 = cosine_similarity_BCL(target_list, label_t_resize, feature_t, feature_s, label_s)

    metric_loss1 = prototype_loss(cosine_similarity_map, label_t.long())
    metric_loss2 = prototype_loss(cosine_similarity_map2, label_s.long())

    metric_loss = config.lamb_metric1 * metric_loss1 + config.lamb_metric2 * metric_loss2
    return metric_loss


def loss_class_prior(pred, prior, w, device):
    prob_pred = pred.mean(dim=(0, 2, 3))
    loss = torch.nn.ReLU()(w * prior - prob_pred)
    return loss.sum()


def exp_func(v1, v2, tau=0.1):
    h = torch.exp((torch.matmul(v1, v2) / tau))
    return h


class ContrastiveLoss(nn.Module):
    def __init__(self, tau=5, n_class=4, bg=False, norm=True):
        super(ContrastiveLoss, self).__init__()
        self._tau = tau
        self._norm = norm
        self._n_class = n_class
        self._bg = bg

    def forward(self, centroid_s, centroid_t, bg=False, split=False):
        centroid_s = centroid_s.to(centroid_t.device)
        if self._norm:
            centroid_s = F.normalize(centroid_s, p=2, dim=1)
            centroid_t = F.normalize(centroid_t, p=2, dim=1)
        exp_mm_st = torch.exp(torch.mm(centroid_t, centroid_s.transpose(0, 1)) / self._tau)
        exp_mm_tt = torch.exp(torch.mm(centroid_t, centroid_t.transpose(0, 1)) / self._tau)
        start_idx = 0 if self._bg else 1
        diag_elements_st = torch.diag(exp_mm_st[start_idx:])
        diag_elements_tt = torch.diag(exp_mm_tt[start_idx:])
        numerator = diag_elements_st + diag_elements_tt
        denominator_st = exp_mm_st[start_idx:].sum(dim=1)
        denominator_tt = exp_mm_tt[start_idx:].sum(dim=1)
        denominator = denominator_st + denominator_tt
        denominator = denominator.clamp(min=1e-7)
        loss_per_class = -torch.log(numerator / denominator)
        loss = loss_per_class.sum()
        return loss


def contrastive_loss(centroid_s, centroid_t, tau=5, n_class=4, bg_included=False, norm=False):
    """
    Deprecated: Replaced by class ContrastiveLoss for cleaner implementation.
    """
    if norm:
        norm_s = torch.norm(centroid_s, p=2, dim=1, keepdim=True)
        norm_t = torch.norm(centroid_t, p=2, dim=1, keepdim=True)
        centroid_s = centroid_s / norm_s
        centroid_t = centroid_t / norm_t
    loss = 0
    for i in range(0 if bg_included else 1, n_class):
        exp_sum = 0
        exp_self = 0
        for j in range(n_class):
            if j != i:
                exp_sum = exp_sum + exp_func(centroid_t[i], centroid_t[j], tau=tau)
        for j in range(n_class):
            if i == j:
                exp_self = exp_func(centroid_t[i], centroid_s[j], tau=tau)
            exp_sum = exp_sum + exp_func(centroid_t[i], centroid_s[j], tau=tau)
        if exp_sum.item() == 0:
            continue
        logit = torch.unsqueeze(torch.unsqueeze(torch.log(torch.div(exp_self, exp_sum)), 0), 0)
        loss_class = nn.NLLLoss()(logit, torch.tensor([0], requires_grad=False).to(centroid_s.device))
        if math.isnan(loss_class.item()):
            print('nan!!!')
        loss = loss + loss_class
    return loss


class SupConLoss(nn.Module):
    """modified supcon loss for segmentation application, the main difference is that the label for different view
    could be different if after spatial transformation"""

    def __init__(self, temperature=0.07,
                 contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):
        device = features.device
        if features.ndim <= 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 4 dimensions are required')
        if features.ndim == 5:
            contrast_count = features.shape[1]
            contrast_feature = torch.cat(torch.unbind(features, dim=1),
                                         dim=0)
        kernels = contrast_feature.permute(0, 2, 3, 1)
        kernels = kernels.reshape(-1, contrast_feature.shape[1], 1, 1)
        logits = torch.div(F.conv2d(contrast_feature, kernels),
                           self.temperature)
        logits = logits.permute(1, 0, 2, 3)
        logits = logits.reshape(logits.shape[0], -1)
        if labels is not None:
            labels = torch.cat(torch.unbind(labels, dim=1), dim=0)
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
            bg_bool = torch.eq(labels.squeeze().cpu(), torch.zeros(labels.squeeze().shape))
            non_bg_bool = ~ bg_bool
            non_bg_bool = non_bg_bool.int().to(device)
        else:
            mask = torch.eye(logits.shape[0] // contrast_count).float().to(device)
            mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(mask.shape[0]).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(
            exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos
        if labels is not None:
            loss = (loss * non_bg_bool).sum() / (non_bg_bool.sum() + 1e-7)
        else:
            loss = loss.mean()
        return loss


class LocalConLoss(nn.Module):
    def __init__(self, temperature=0.7, stride=4):
        super(LocalConLoss, self).__init__()
        self.temp = temperature
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.supconloss = SupConLoss(temperature=self.temp)
        self.stride = stride

    def forward(self, features, labels=None):
        features = features[:, :, :, ::self.stride, ::self.stride]
        if labels is not None:
            labels = labels[:, :, ::self.stride, ::self.stride]
            if labels.sum() == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss
            loss = self.supconloss(features, labels)
            return loss
        else:
            loss = self.supconloss(features)
            return loss


class BlockConLoss(nn.Module):
    def __init__(self, temperature=0.7, block_size=32):
        super(BlockConLoss, self).__init__()
        self.block_size = block_size
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.supconloss = SupConLoss(temperature=temperature)

    def forward(self, features, labels=None):
        shape = features.shape
        img_size = shape[-1]
        div_num = img_size // self.block_size
        if labels is not None:
            loss = []
            for i in range(div_num):
                for j in range(div_num):
                    block_features = features[:, :, :, i * self.block_size:(i + 1) * self.block_size,
                                              j * self.block_size:(j + 1) * self.block_size]
                    block_labels = labels[:, :, i * self.block_size:(i + 1) * self.block_size,
                                   j * self.block_size:(j + 1) * self.block_size]
                    if block_labels.sum() == 0:
                        continue
                    tmp_loss = self.supconloss(block_features, block_labels)
                    loss.append(tmp_loss)
            if len(loss) == 0:
                loss = torch.tensor(0).float().to(self.device)
                return loss
            loss = torch.stack(loss).mean()
            return loss
        else:
            loss = []
            for i in range(div_num):
                for j in range(div_num):
                    block_features = features[:, :, :, i * self.block_size:(i + 1) * self.block_size,
                                              j * self.block_size:(j + 1) * self.block_size]
                    tmp_loss = self.supconloss(block_features)
                    loss.append(tmp_loss)
            loss = torch.stack(loss).mean()
            return loss


class MPCL(nn.Module):
    def __init__(self, device, num_class=5, temperature=0.07, m=0.5,
                 base_temperature=0.07, easy_margin=False):
        super(MPCL, self).__init__()
        self.num_class = num_class
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.device = device
        self.easy_margin = easy_margin

    def forward(self, features, labels, class_center_feas,
                pixel_sel_loc=None, mask=None):
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        num_samples = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(num_samples, dtype=torch.float32, device=self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1).long().to(self.device)
            class_center_labels = torch.arange(0, self.num_class).long().to(self.device)
            class_center_labels = class_center_labels.contiguous().view(-1, 1)
            if labels.shape[0] != num_samples:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, torch.transpose(class_center_labels, 0, 1)).float().to(self.device)
        else:
            mask = mask.float().to(self.device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        cosine = torch.matmul(anchor_feature, class_center_feas)
        logits = torch.div(cosine, self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0.0001, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        phi_logits = torch.div(phi, self.temperature)
        phi_logits_max, _ = torch.max(phi_logits, dim=1, keepdim=True)
        phi_logits = phi_logits - phi_logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        tag_1 = (1 - mask)
        tag_2 = mask
        exp_logits = torch.exp(logits * tag_1 + phi_logits * tag_2)
        phi_logits = (logits * tag_1) + (phi_logits * tag_2)
        log_prob = phi_logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-4)
        if pixel_sel_loc is not None:
            pixel_sel_loc = pixel_sel_loc.view(-1)
            mean_log_prob_pos = (mask * log_prob).sum(1)
            mean_log_prob_pos = pixel_sel_loc * mean_log_prob_pos
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = torch.div(loss.sum(), pixel_sel_loc.sum() + 1e-4)
        else:
            mean_log_prob_pos = (mask * log_prob).sum(1)
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.view(anchor_count, num_samples).mean()
        return loss


def mpcl_loss_calc(feas, labels, class_center_feas, loss_func,
                   pixel_sel_loc=None, tag='source'):
    n, c, fea_h, fea_w = feas.size()
    if tag == 'source' and (labels.size()[1] != fea_h or labels.size()[2] != fea_w):
        labels = labels.float()
        labels = F.interpolate(labels.unsqueeze(1), size=(fea_h, fea_w), mode='nearest').squeeze(1)
    labels = labels.to(feas.device)
    labels = labels.view(-1).long()
    feas = torch.nn.functional.normalize(feas, p=2, dim=1)
    feas = feas.transpose(1, 2).transpose(2, 3).contiguous()
    feas = torch.reshape(feas, [n * fea_h * fea_w, c])
    feas = feas.unsqueeze(1)
    class_center_feas = torch.nn.functional.normalize(class_center_feas, p=2, dim=1)
    class_center_feas = torch.transpose(class_center_feas, 0, 1)
    loss = loss_func(feas, labels, class_center_feas,
                     pixel_sel_loc=pixel_sel_loc)
    return loss


def batch_pairwise_dist(x, y):
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points, device=x.device).long()
    rx = xx[:, diag_ind, diag_ind]
    rx = rx.unsqueeze(1)
    rx = rx.expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz).clamp(min=0)
    return P


def batch_NN_loss(x, y):
    smooth = 1e-7
    bs, num_points, points_dim = x.size()
    dist1 = torch.sqrt(batch_pairwise_dist(x, y) + smooth)
    values1, indices1 = dist1.min(dim=2)
    dist2 = torch.sqrt(batch_pairwise_dist(y, x) + smooth)
    values2, indices2 = dist2.min(dim=2)
    a = torch.div(torch.sum(values1, 1), num_points)
    b = torch.div(torch.sum(values2, 1), num_points)
    sum = torch.div(torch.sum(a), bs) + torch.div(torch.sum(b), bs)
    return sum