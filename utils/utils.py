import os
import faiss
import numpy as np
from tqdm import tqdm


def compute_ece(preds, variances, gt):
    '''return: bins_recall: (10, 3)
                ece_bins_recall: (1, 3)    
    '''
    assert len(preds) == len(variances) == len(gt), 'length of preds, variances and gt should be the same :('
    num_bins = 11
    n_values = [1, 5, 10]
    if variances.shape[-1] != 1:
        variances = variances.mean(axis=-1)
    variance_reduced = variances
    indices, _ = get_bins(variance_reduced, num_bins)
    bins_recall = np.zeros((num_bins - 1, len(n_values)))
    ece_bins_recall = np.zeros((num_bins - 1, len(n_values)))
    for index in range(num_bins - 1):
        if len(indices[index]) == 0:
            continue
        pred_bin = preds[indices[index]]
        gt_bin = [gt[i] for i in indices[index]]
        # calculate r@K
        recall_at_n = cal_recall(pred_bin, gt_bin, n_values)
        bins_recall[index] = recall_at_n
        ece_bins_recall[index] = np.array([len(indices[index]) / variance_reduced.shape[0] * np.abs(recall_at_n[i] - (num_bins - 1 - index) / ((num_bins - 1))) for i in range(len(n_values))], )
    return bins_recall, ece_bins_recall.sum(axis=0), np.array([len(x) for x in indices])


def schedule_device():
    info_per_card = os.popen('"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().split('\n')

    card_memory_used = []
    for i in range(len(info_per_card)):
        if info_per_card[i] == '':
            continue
        else:
            total, used = int(info_per_card[i].split(',')[0]), int(info_per_card[i].split(',')[1])
            card_memory_used.append(used)
    return int(card_memory_used.index(min(card_memory_used)))


def find_nn(q_mu, db_mu, num_nn, device=''):
    """
    retrieve by L2 distance:||x-y||^2
    """
    if device == '':
        if q_mu.shape[0] >= 10000:
            device = 'gpu'
        else:
            device = 'cpu'

    if device == 'gpu':
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = 2
        faiss_index = faiss.GpuIndexFlatL2(res, q_mu.shape[1], flat_config)
    elif device == 'cpu':
        faiss_index = faiss.IndexFlatL2(q_mu.shape[1])
    faiss_index.add(db_mu)
    dists, preds = faiss_index.search(q_mu, num_nn)
    return dists, preds


def cal_recall(ranks, pidx, ks):
    recall_at_k = np.zeros(len(ks))
    for qidx in range(ranks.shape[0]):
        for i, k in enumerate(ks):
            if np.sum(np.in1d(ranks[qidx, :k], pidx[qidx])) > 0:
                recall_at_k[i:] += 1
                break
    recall_at_k /= ranks.shape[0]
    return recall_at_k


def get_bins(q_sigma_sq_h, num_of_bins):
    q_sigma_sq_h_min = np.min(q_sigma_sq_h)
    q_sigma_sq_h_max = np.max(q_sigma_sq_h)
    bins = np.linspace(q_sigma_sq_h_min, q_sigma_sq_h_max, num=num_of_bins)
    indices = []
    for index in range(num_of_bins - 1):
        target_q_ind_l = np.where(q_sigma_sq_h >= bins[index])
        if index != num_of_bins - 2:
            target_q_ind_r = np.where(q_sigma_sq_h < bins[index + 1])
        else:
            # the last bin use close interval
            target_q_ind_r = np.where(q_sigma_sq_h <= bins[index + 1])

        target_q_ind = np.intersect1d(target_q_ind_l, target_q_ind_r)
        indices.append(target_q_ind)
    return indices, bins
