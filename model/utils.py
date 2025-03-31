import os
import torch
from sklearn.metrics import accuracy_score

class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(model, optimizer, logdir, epoch, global_step):
    torch.save(
        {
            "epoch": epoch,
            "global_iter": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(
            logdir,
            "checkpoints",
            "epoch_{}_globaliter_{}".format(epoch, global_step) + ".pt",
        ),
    )


def BinaryAcc(preds, label, masks, all_pred, all_true):
    """ return: all_pred, all_true list, per-batch accuracy """
    A_spk_p, B_spk_p, A_lst_p, B_lst_p = preds
    A_spk_t, B_spk_t, A_lst_t, B_lst_t = label
    A_spk_m, B_spk_m, A_lst_m, B_lst_m = masks

    A_spk_p = torch.argmax(A_spk_p, dim=1)
    B_spk_p = torch.argmax(B_spk_p, dim=1)
    A_lst_p = torch.argmax(A_lst_p, dim=1)
    B_lst_p = torch.argmax(B_lst_p, dim=1)

    A_spk_acc = accuracy_score(A_spk_p.cpu(), A_spk_t.cpu())
    B_spk_acc = accuracy_score(B_spk_p.cpu(), B_spk_t.cpu())
    A_lst_acc = accuracy_score(A_lst_p.cpu(), A_lst_t.cpu())
    B_lst_acc = accuracy_score(B_lst_p.cpu(), B_lst_t.cpu())

    A_spk_val_idx = (A_spk_m == 1).nonzero()
    B_spk_val_idx = (B_spk_m == 1).nonzero()
    A_lst_val_idx = (A_lst_m == 1).nonzero()
    B_lst_val_idx = (B_lst_m == 1).nonzero()

    A_spk_all_pred, B_spk_all_pred, A_lst_all_pred, B_lst_all_pred = all_pred
    A_spk_all_true, B_spk_all_true, A_lst_all_true, B_lst_all_true = all_true

    for i in A_spk_val_idx:
        A_spk_all_pred.append(A_spk_p[i].cpu())
        A_spk_all_true.append(A_spk_t[i].cpu())

    for i in B_spk_val_idx:
        B_spk_all_pred.append(B_spk_p[i].cpu())
        B_spk_all_true.append(B_spk_t[i].cpu())

    for i in A_lst_val_idx:
        A_lst_all_pred.append(A_lst_p[i].cpu())
        A_lst_all_true.append(A_lst_t[i].cpu())

    for i in B_lst_val_idx:
        B_lst_all_pred.append(B_lst_p[i].cpu())
        B_lst_all_true.append(B_lst_t[i].cpu())

    output_pred = [A_spk_all_pred, B_spk_all_pred, A_lst_all_pred, B_lst_all_pred]
    output_true = [A_spk_all_true, B_spk_all_true, A_lst_all_true, B_lst_all_true]
    output_accs = [A_spk_acc, B_spk_acc, A_lst_acc, B_lst_acc]

    return output_pred, output_true, output_accs


def PerTaskLabel(rels):
    edge, mask = rels
    if len(edge.shape) == 3:
        edge = edge.unsqueeze(0)
        mask = mask.unsqueeze(0)

    # Transpose edge and mask to get per task labels
    if len(edge.shape) == 4:
        # [B, T, N, K] -> [B, N, T, K] -> [B*N, T, K] -> [K, T, B*N]
        B, T, N, K = edge.shape
        edge = edge.permute(0, 2, 1, 3).reshape(B * N, T, K)
        edge = edge.permute(2, 1, 0)
    elif len(edge.shape) == 3:
        # [B, N, K] -> [B*N, K] -> [K, B*N]
        B, N, _ = edge.shape
        edge = edge.view(B * N, -1)
        edge = edge.permute(1, 0)

    if len(mask.shape) == 4:
        # [B, T, N, K] -> [B, N, T, K] -> [B*N, T, K] -> [K, T, B*N]
        B, T, N, K = mask.shape
        mask = mask.permute(0, 2, 1, 3).reshape(B * N, T, K)
        mask = mask.permute(2, 1, 0)
    elif len(mask.shape) == 3:
        # [B, N, K] -> [B*N, K] -> [K, B*N]
        mask = mask.view(B * N, -1)
        mask = mask.permute(1, 0)

    A_spk_label = edge[0].T.reshape(B * N * T)
    B_spk_label = edge[1].T.reshape(B * N * T)
    A_lst_label = edge[2].T.reshape(B * N * T)
    B_lst_label = edge[3].T.reshape(B * N * T)

    A_spk_mask = mask[0].T.reshape(B * N * T)
    B_spk_mask = mask[1].T.reshape(B * N * T)
    A_lst_mask = mask[2].T.reshape(B * N * T)
    B_lst_mask = mask[3].T.reshape(B * N * T)

    label = [A_spk_label, B_spk_label, A_lst_label, B_lst_label]
    mask  = [A_spk_mask, B_spk_mask, A_lst_mask, B_lst_mask]

    return label, mask
