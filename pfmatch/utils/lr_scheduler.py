from __future__ import annotations

import torch

def scheduler_factory(opt : torch.optim.Optimizer,
                      cfg : dict, 
                      verbose=False):
    '''
    Learning rate factory.

    Arguments
    ---------
    opt: Optimizer
        A `torch.optim.Optimizer` instance.

    cfg: dict
        Configuration dictory

        `Class`: str
            class name of the scheduler from `torch.optim.lr_scheduler`.

        `Param`: dict
            keyword argumets passed to the scheduler constructor.

    Returns
    -------
    sch: Scheduler
        An instance of the scheduler.
        `None` if `cfg == None`, or `Class` key not found in `cfg`.
    '''
    if cfg is None:
        return None

    sch_class = cfg.get('Class')
    sch_param = cfg.get('Param', {})

    if sch_class is None:
        return None

    if not hasattr(torch.optim.lr_scheduler, sch_class):
        raise RuntimeError(
            f'torch.optim.lr_scheduler has no scheduler called {sch_class}'
        )

    sch = getattr(torch.optim.lr_scheduler, sch_class)(opt, **sch_param)

    if verbose:
        print('[lr_scheduler]', sch_class)
        print('[lr_scheduler]', sch_param)

    return sch
