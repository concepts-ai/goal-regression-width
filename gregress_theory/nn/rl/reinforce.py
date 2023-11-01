
import torch.nn as nn

__all__ = ['REINFORCELoss']


class REINFORCELoss(nn.Module):
    def __init__(self, entropy_beta=None):
        super().__init__()
        self.nll = nn.NLLLoss(reduce=False)
        self.entropy_beta = entropy_beta

    def forward(self, policy, action, discount_reward, entropy_beta=None):
        monitors = dict()
        entropy = -(policy * policy.log()).sum(dim=1).mean()
        nll = self.nll(policy, action)
        loss = (nll * discount_reward).mean()
        if entropy_beta is None:
            entropy_beta = self.entropy_beta
        if entropy_beta is not None:
            monitors['reinforce_loss'] = loss
            monitors['entropy_loss'] = -entropy * entropy_beta
            loss -= entropy * entropy_beta
        monitors['entropy'] = entropy
        return loss, monitors
