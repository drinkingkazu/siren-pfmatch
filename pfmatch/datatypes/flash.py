import copy
import torch
import numpy as np

class Flash:
    """
    A single Flash contains a list of PE values (pe_v) and a list of PE error values (pe_err_v) for each PMT.
    In ICARUS, there are 180 PMTs so pe_v and pe_err_v are both of shape (180,).
    """
    
    def __init__(self, pe_v, pe_true_v=None, pe_err_v=None,
        idx=2**31-1, time=np.inf, time_true=np.inf, time_width=np.inf):
 
        self._pe_v = torch.as_tensor(pe_v)
        self._pe_true_v = pe_true_v if pe_true_v is None else torch.as_tensor(pe_true_v)
        self._pe_err_v  = pe_err_v  if pe_err_v  is None else torch.as_tensor(pe_err_v )

        self._idx = idx        # index from original larlite vector
        self._time = time      # Flash timing, a candidate T0
        self._time_true = time_true   # MCFlash timing
        self._time_width = time_width # flash time integration window

    def __len__(self):
        return len(self.pe_v)

    def sum(self):
        if len(self.pe_v) == 0:
            return 0
        return torch.sum(self.pe_v).item()

    @property
    def pe_v(self):
        return self._pe_v

    @property
    def pe_true_v(self):
        return self._pe_true_v

    @property
    def pe_err_v(self):
        return self._pe_err_v

    @property
    def idx(self):
        return self._idx
    
    @property
    def time(self):
        return self._time

    @property
    def time_true(self):
        return self._time_true
    
    @property
    def time_width(self):
        return self._time_width