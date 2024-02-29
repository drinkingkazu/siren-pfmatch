import torch

class PoissonMatchLoss(torch.nn.Module):
    """
    Poisson NLL Loss for gradient-based optimization model
    """
    def __init__(self):
        super(PoissonMatchLoss, self).__init__()
        self.poisson_nll = torch.nn.PoissonNLLLoss(log_input=False,
            full=True,
            reduction="none")

    def forward(self, input, target, weight=1., axis=-1):
        H = torch.clamp(input, min=0.01)
        O = torch.clamp(target, min=0.01)
        loss = self.poisson_nll(H, O) - torch.log(H) / 2
        return torch.mean(weight*loss, axis=axis)
