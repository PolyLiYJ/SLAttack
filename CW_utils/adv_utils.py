import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitsAdvLoss(nn.Module):

    def __init__(self, kappa=0., whether_target= False, device=torch.device("cpu")):
        """Adversarial function on logits.
        Args:
            kappa (float, optional): min margin. Defaults to 0..
        """
        super(LogitsAdvLoss, self).__init__()

        self.kappa = kappa
        self.whether_target = whether_target
        self.device = device

    def forward(self, logits, labels):
        """Adversarial loss function using logits.
        Args:
            logits (torch.FloatTensor): output logits from network, [B, K]
            targets (torch.LongTensor): attack target class
        """
        B, K = logits.shape
        if self.whether_target:
            targets = labels
            if len(targets.shape) == 1:
                targets = targets.view(-1, 1)
            targets = targets.long()
            one_hot_targets = torch.zeros(B, K).to(self.device).scatter_(
                1, targets, 1).float()  # to one-hot
            real_logits = torch.sum(one_hot_targets * logits, dim=1)
            other_logits = torch.max((1. - one_hot_targets) * logits -
                                    one_hot_targets * 10000., dim=1)[0]
            loss = torch.clamp(other_logits - real_logits + self.kappa, min=0.)
        else:
            ori = labels
            if len(ori.shape) == 1:
                ori = ori.view(-1, 1)
            ori = ori.long()
            one_hot_ori = torch.zeros(B, K).to(self.device).scatter_(
                1, ori, 1).float()  # to one-hot
            real_logits = torch.sum(one_hot_ori * logits, dim=1)
            other_logits = torch.max((1. - one_hot_ori) * logits -
                                    one_hot_ori * 10000., dim=1)[0]
            loss = torch.clamp(real_logits-other_logits + self.kappa, min=0.)
        return loss.mean()


class CrossEntropyAdvLoss(nn.Module):

    def __init__(self):
        """Adversarial function on output probabilities.
        """
        super(CrossEntropyAdvLoss, self).__init__()

    def forward(self, logits, targets):
        """Adversarial loss function using cross entropy.
        Args:
            logits (torch.FloatTensor): output logits from network, [B, K]
            targets (torch.LongTensor): attack target class
        """
        # loss = F.cross_entropy(logits, targets)
        loss = F.nll_loss(logits, targets)
        return loss
