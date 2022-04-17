import bottleneck as bn
import numpy as np
import torch


def ndcg(X_pred, heldout_batch, k=100, is_hvamp=False):
    """
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    """
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    if is_hvamp:
        tp = torch.tensor(tp, dtype=torch.float)
        DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                             idx_topk].cpu() * tp).sum(dim=1)
        IDCG = torch.tensor([(tp[:min(n, k)]).sum()
                             for n in (heldout_batch != 0).sum(dim=1)])
    else:
        DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                             idx_topk].toarray() * tp).sum(axis=1)
        IDCG = np.array([(tp[:min(n, k)]).sum()
                         for n in heldout_batch.getnnz(axis=1)])

    return DCG / IDCG


def recall(X_pred, heldout_batch, k=100, is_hvamp=False):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True


    if is_hvamp:
        X_true_binary = (heldout_batch > 0).clone().detach()
        X_true_binary = X_true_binary.cpu().detach()

        tmp = torch.tensor(np.logical_and(X_true_binary.numpy(), X_pred_binary), dtype=torch.float).sum(dim=1)
        recall_value = tmp / np.minimum(k, X_true_binary.cpu().detach().sum(dim=1))

    else:
        X_true_binary = (heldout_batch > 0).toarray()

        tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
            np.float32)
        recall_value = tmp / np.minimum(k, X_true_binary.sum(axis=1))

    return recall_value
