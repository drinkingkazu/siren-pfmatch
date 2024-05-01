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
        return torch.nanmean(weight*loss, axis=axis)

class Chi2Loss(torch.nn.Module):
    '''
    Chi2 loss w/ additional (constant) error terms.
    '''
    def __init__(self, err0=0):
        super().__init__()
        self.register_buffer(
            '_err0', torch.as_tensor(err0, dtype=torch.float32)
        )
        
    def forward(self, input, target):
        w = 1 / (input + self._err0**2)
        mask = ~target.isnan()
        loss = w[mask] * (input[mask] - target[mask])**2
        return loss.mean()

    def __str__(self):
        return f'Chi2Loss(err0={self._err0})'
