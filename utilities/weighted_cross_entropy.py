import torch
import torch.nn.functional as F


def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    # calculating label weights for weighted loss computation
    true = true.to(torch.long)
    V = true.size(0)                                                            # batch size
    n_classes = pred.shape[1] if pred.ndim > 1 else 2                           # number of classes
    label_count = torch.bincount(true)                                          # count the frequency of each class
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()     # removes any zero counts
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()           
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V                                     # calculates the weights for each class
    weight *= (cluster_sizes > 0).float()                                        # only non-zero cluster sizes contribute to the weight
    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        return F.nll_loss(pred, true, weight=weight), pred
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(pred, true.float(), weight=weight[true])
        return loss, torch.sigmoid(pred)