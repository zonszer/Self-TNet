import torch
import torch.nn.functional as F
import numpy as np
from numpy import polynomial as P
import torch.nn as nn
import random
eps_sqrt = 1e-6
from info_nce import InfoNCE

def distance_matrix_vector(anchor, positive, eps = 1e-6, detach_other=False):
    """# Given batch of anchor descriptors and positive descriptors calculate distance matrix"""
    d1_sq = torch.norm(anchor, p=2, dim=1, keepdim = True)
    d2_sq = torch.norm(positive, p=2, dim=1, keepdim = True)
    if detach_other:
        d2_sq = d2_sq.detach() # -> negatives do not get gradient
    # descriptors are still normalized, this is just more general formula
    return torch.sqrt(d1_sq.repeat(1, positive.size(0))**2 + torch.t(d2_sq.repeat(1, anchor.size(0)))**2 - 2.0 * F.linear(anchor, positive) + eps)


def tripletMargin_generalized(embeddings, labels,  # this is general implementation with embeds+labels and is fast
                              margin_pos=1.0,  # if edge higher than margin_pos, change
                              ):
    N = len(labels)
    with torch.no_grad():
        dm = torch.cdist(embeddings, embeddings)
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).float()
        # is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
        HUGE_CONST = 1e5
        pos_dist_max, pos_distmax_idx = (is_pos * dm).max(dim=1)
        too_close = dm.le(0.008).float()
        # neg_dist_min, neg_distmin_idx = (HUGE_CONST * is_pos.float() + dm).min(dim=1)
        neg_dist_min, neg_distmin_idx = (is_pos*HUGE_CONST + too_close*HUGE_CONST + dm).min(dim=1)

        labels_unique = torch.unique(labels)
        NL = len(labels_unique)
        label_class_matrix = labels.unsqueeze(0).expand(NL, N).eq(labels_unique.unsqueeze(0).expand(N, NL).t())

        pos_dist_max_class = (pos_dist_max.unsqueeze(0).expand(NL, N) * label_class_matrix.float()).max(dim=1)
        neg_dist_min_class = (neg_dist_min.unsqueeze(0).expand(NL, N) + HUGE_CONST * (~label_class_matrix).float()).min(dim=1)

    a = torch.norm(embeddings[pos_dist_max_class[1]] - embeddings[pos_distmax_idx[pos_dist_max_class[1]]], dim=1)
    b = torch.norm(embeddings[neg_dist_min_class[1]] - embeddings[neg_distmin_idx[neg_dist_min_class[1]]], dim=1)
    edge = a - b
    loss = torch.clamp(margin_pos + edge, min=0.0)
    return loss.mean(), (a.detach(), b.detach(), edge.mean())           #


def cal_l2_distance_matrix(x, y, flag_sqrt=True):
    ''''distance matrix of x with respect to y, d_ij is the distance between x_i and y_j'''
    D = torch.abs(2 * (1 - torch.mm(x, y.t())))    #D is Dl2.pow(2)
    if flag_sqrt:
        D = torch.sqrt(D + eps_sqrt)
    return D


def tripletMargin_generalized_Exponential(embeddings, labels,  # this is general implementation with embeds+labels and is fast
                              margin_pos=1.0,  # if edge higher than margin_pos, change
                              neg_num=1,
                              eps = 1e-8,
                                R = 1.5,
                                B = 3.8,
                                use_stB=False,
                                A = 1.0,
                                D=0.83, C=0.0,
                                threshold=-0.15, is_finetune=False,
                                ranges= 0.2
                              ):
    N = len(labels)
    with torch.no_grad():
        dm = cal_l2_distance_matrix(embeddings, embeddings)    #compute the L2 dist.pow(2) and return matrix
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).float()

        pos_dist_max, pos_distmax_idx = (is_pos * dm).max(dim=1)

        labels_unique = torch.unique(labels)
        NL = len(labels_unique)
        label_class_matrix = labels.unsqueeze(0).expand(NL, N).eq( labels_unique.unsqueeze(0).expand(N, NL).t() )

        pos_dist_max_class = (pos_dist_max.unsqueeze(0).expand(NL, N) * label_class_matrix.float()).max(dim=1)
        HUGE_CONST = 1e5

        too_close = dm.le(0.008).float()
        filtered_dm = is_pos * HUGE_CONST + too_close * HUGE_CONST + dm
        neg_dist_min_class_list = []
        neg_dist_min, neg_distmin_idx = filtered_dm.min(dim=1)  # horizontial min value, neg_idx is col idx
        neg_dist_min_class = (neg_dist_min.unsqueeze(0).expand(NL, N) + HUGE_CONST * (~label_class_matrix).float()).min(dim=1)

        neg_dist_min_class_list.append((neg_dist_min_class[1], neg_distmin_idx[neg_dist_min_class[1]]))
        no_repeatElem_row = neg_dist_min_class[1]
        for i in range(neg_num - 1):
            filtered_dm[no_repeatElem_row, neg_dist_min_class[1]] = HUGE_CONST
            neg_dist_min_class = filtered_dm[no_repeatElem_row].min(dim=1)
            neg_dist_min_class_list.append((neg_dist_min_class[1], neg_distmin_idx[neg_dist_min_class[1]]))
    a = torch.norm(embeddings[pos_dist_max_class[1]] - embeddings[pos_distmax_idx[pos_dist_max_class[1]]], dim=1) 
    sum_b = torch.zeros(NL).cuda()

    for idx, (i, j) in enumerate( neg_dist_min_class_list):
        b = torch.norm(embeddings[i] - embeddings[j], dim=1)
        if use_stB:
            # assert neg_num == 1
            B = (b.detach().median() + a.detach().median() * A) * 2
        b = torch.clamp(b, max=B/2)
        if is_finetune:
            confident_weight = (b - a > threshold).float()      #true_label
            a = confident_weight * a
            b = confident_weight * b
        else:
            confident_weight = 1
        sum_b = (torch.pow(b, 2) - B*b) + sum_b

    loss = R * (torch.pow(a, 2)) + sum_b / neg_num
    loss = loss + 5
    return loss.mean(), (a.detach(), b.detach(), B, C, confident_weight)  


def generate_weight(a, b, threshold, upper=-0.1):
    confident_level = b - a
    true_label = (confident_level > threshold).float() * (confident_level < upper).float()

    m = -1/(threshold - upper)** 2       #thr and upper all be neg, so make them positive

    clumped_value = confident_level * true_label + torch.abs(true_label - 1) * upper
    confident_weight = m*(clumped_value - upper)** 2 + 1 

    no_w0_label = (confident_level > threshold).float()
    confident_weight = confident_weight * no_w0_label
    return confident_weight


def tripletMargin_generalized_ExpTeacher(used_neg_dist, embeddings, embeddingsT, labels,
                                R = 1.5,
                                B = 3.8,
                                use_stB=False,
                                A = 1.0,
                                C=0.0,
                                threshold=-0.15,
                                upper=-0.1
                              ):
    """
    Calculate the loss for a generalized triplet margin with exponential teacher.

    Args:
        used_neg_dist (Tensor): Used negative distances.
        embeddings (Tensor): Embeddings of the current batch.
        embeddingsT (Tensor): Embeddings of the teacher model.
        labels (Tensor): Labels of the current batch.
        R (float, optional): Scaling factor for the positive distance.
        B (float, optional): Maximum allowed negative distance.
        use_stB (bool, optional): Whether to use a dynamic B calculated from data.
        A (float, optional): Scaling factor for the positive distance in dynamic B.
        C (float, optional): Constant added to the loss.
        threshold (float, optional): Threshold for the confident weight.
        upper (float, optional): Upper limit for the confident weight.

    Returns:
        tuple: The mean loss and a tuple with the positive distance, negative distance, B, C and confident weight.
    """
    N = len(labels)
    with torch.no_grad():
        dm = cal_l2_distance_matrix(embeddings, embeddings)    #compute the L2 dist.pow(2) and return matrix

        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).float()
        HUGE_CONST = 1e5
        pos_dist_max, pos_distmax_idx = (is_pos * dm).max(dim=1)
        too_close = dm.le(0.008).float()
        # neg_dist_min, neg_distmin_idx = (HUGE_CONST * is_pos.float() + dm).min(dim=1)
        neg_dist_min, neg_distmin_idx = (is_pos*HUGE_CONST + too_close*HUGE_CONST + used_neg_dist*HUGE_CONST + dm).min(dim=1)  # horizontial min value, neg_idx is col idx

        used_neg_dist[ torch.arange(N) ,neg_distmin_idx ] = 1

        labels_unique = torch.unique(labels)
        NL = len(labels_unique) 
        label_class_matrix = labels.unsqueeze(0).expand(NL, N).eq(labels_unique.unsqueeze(0).expand(N, NL).t())
        # functions of label_class_matrix is to get the label of the same class
        pos_dist_max_class = (pos_dist_max.unsqueeze(0).expand(NL, N) * label_class_matrix.float()).max(dim=1)

        neg_dist_min_class = (neg_dist_min.unsqueeze(0).expand(NL, N) + HUGE_CONST * (~label_class_matrix).float()).min(dim=1)
        a_T = torch.norm(embeddingsT[pos_dist_max_class[1]] - embeddingsT[pos_distmax_idx[pos_dist_max_class[1]]], dim=1)
        b_T = torch.norm(embeddingsT[neg_dist_min_class[1]] - embeddingsT[neg_distmin_idx[neg_dist_min_class[1]]], dim=1)
        # confident_weight = generate_weight(a_T, b_T, threshold, upper)
        if threshold == upper:
            confident_weight = 1
        else:
            confident_weight = generate_weight(a_T, b_T, threshold, upper)

    a = torch.norm(embeddings[pos_dist_max_class[1]] - embeddings[pos_distmax_idx[pos_dist_max_class[1]]], dim=1)
    b = torch.norm(embeddings[neg_dist_min_class[1]] - embeddings[neg_distmin_idx[neg_dist_min_class[1]]], dim=1)

    if use_stB:
        B = (b.detach().median() + a.detach().median() * A) * 2

    b = torch.clamp(b, max=B/2)
    sum_b = (torch.pow(b, 2) - B*b)
    loss = confident_weight * (R * (torch.pow(a, 2)) + sum_b) + 5
    return loss.mean(), (a.detach(), b.detach(), B, C, confident_weight)



def get_indicator(mu, speedup=10.0, type='le'): # differentiable indicator, returns 1 if input < mu
    assert type in ['le', 'ge']
    if type in ['le']:
        return lambda x : indicator_le(x, mu, speedup)
    elif type in ['ge']:
        return lambda x : indicator_ge(x, mu, speedup)
def indicator_le(input, mu, speedup=10.0): # differentiable indicator, returns 1 if input < mu
    x = -(input - mu) # -input flips by y-axis, -mu shifts by value
    return torch.sigmoid(x * speedup)
def indicator_ge(input, mu, speedup=10.0): # differentiable indicator, returns 1 if input > mu
    x = (input - mu) # -input flips by y-axis, -mu shifts by value
    return torch.sigmoid(x * speedup)

def indicator_le(input, speedup=10.0): # differentiable indicator, returns 1 if input < mu
    return torch.sigmoid(-input * speedup) # -input flips by y-axis, -mu shifts by value
def indicator_ge(input, speedup=10.0): # differentiable indicator, returns 1 if input > mu
    return torch.sigmoid(input * speedup) # -input flips by y-axis, -mu shifts by value

# HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
def loss_AP(anchor, positive, eps = 1e-8): # we want sthing like this, with grads
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    dist_matrix = distance_matrix_vector(anchor, positive) + eps

    ss = torch.sort(dist_matrix)[0]
    dd = torch.diag(dist_matrix).view(-1,1)
    hots = (ss-dd)==0
    res = hots.nonzero()[:, 1].float()
    loss = torch.sum(torch.log(res+1))
    return loss

def loss_AP_diff(anchor, positive, speedup, eps = 1e-8): # using appwoximation of indicator
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    dist_matrix = distance_matrix_vector(anchor, positive) + eps

    dist_matrix = dist_matrix - torch.diag(dist_matrix)
    # for i in range(dist_matrix.shape[0]):
    #     f = get_indicator(dist_matrix[i,i], speedup=10.0, type='le')
    #     dist_matrix[i,:] = f(dist_matrix[i,:])
    dist_matrix = indicator_le(dist_matrix, speedup=speedup)

    loss = torch.sum(dist_matrix, dim=1)
    loss = torch.log(loss)
    return loss


def compute_distance_matrix_unit_l2(a, b, eps=1e-6):
    """computes pairwise Euclidean distance and return a N x N matrix"""
    dmat = torch.matmul(a, torch.transpose(b, 0, 1))
    dmat = ((1.0 - dmat + eps) * 2.0).pow(0.5)
    return dmat

def compute_distance_matrix_hamming(a, b):
    """computes pairwise Hamming distance and return a N x N matrix"""
    dims = a.size(1)
    dmat = torch.matmul(a, torch.transpose(b, 0, 1))
    dmat = (dims - dmat) * 0.5
    return dmat

def find_hard_negatives(dmat, output_index=True, empirical_thresh=0.0):
    """a = A * P'
    A: N * ndim
    P: N * ndim

    a1p1 a1p2 a1p3 a1p4 ...
    a2p1 a2p2 a2p3 a2p4 ...
    a3p1 a3p2 a3p3 a3p4 ...
    a4p1 a4p2 a4p3 a4p4 ...
    ...  ...  ...  ..."""
    cnt = dmat.size(0)

    if not output_index:
        pos = dmat.diag()

    dmat = dmat + torch.eye(cnt).to(dmat.device) * 99999  # filter diagonal
    dmat[dmat < empirical_thresh] = 99999  # filter outliers in brown dataset
    min_a, min_a_idx = torch.min(dmat, dim=0)
    min_p, min_p_idx = torch.min(dmat, dim=1)

    if not output_index:
        neg = torch.min(min_a, min_p)
        return pos, neg

    mask = min_a < min_p
    a_idx = torch.cat(
        (mask.nonzero().view(-1) + cnt, (~mask).nonzero().view(-1))
    )  # use p as anchor
    p_idx = torch.cat(
        (mask.nonzero().view(-1), (~mask).nonzero().view(-1) + cnt)
    )  # use a as anchor
    n_idx = torch.cat((min_a_idx[mask], min_p_idx[~mask] + cnt))
    return a_idx, p_idx, n_idx
