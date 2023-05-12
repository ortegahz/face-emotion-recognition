import torch


def label_smooth(target, n_classes: int, label_smoothing=0.1):
    """

    :param target: shape(bs,)
    :param n_classes:
    :param label_smoothing:
    :return:
    """
    # convert to one-hot
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)  # shape(bs, 1)
    soft_target = torch.zeros((batch_size, n_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return soft_target


def cross_entropy_loss_with_soft_target(pred, soft_target, weights):
    # logsoftmax = nn.LogSoftmax(dim=-1)
    return torch.mean(torch.sum(- weights * soft_target * torch.nn.functional.log_softmax(pred, -1), 1))


def cross_entropy_with_label_smoothing(pred, target, weights):
    soft_target = label_smooth(target, pred.size(1))  # num_classes
    return cross_entropy_loss_with_soft_target(pred, soft_target, weights)
