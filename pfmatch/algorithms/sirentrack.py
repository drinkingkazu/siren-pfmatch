import numpy as np
import torch
from slar.nets import SirenVis

class SirenTrack(SirenVis):
    """Siren class implementation for p.e. reconstruction of an entire track in LArTPCs"""

    def __init__(self, cfg : dict, ckpt_file : str = None):
        # this works with siren's that are already trained with voxel data!
        super().__init__(cfg, ckpt_file)
    
    def _inv_xform_vis(self, vis):
        """SirenVis._inv_xform_vis()"""
        return super()._inv_xform_vis(vis)
    
    def forward(self, batch):
        """Forward pass of the SirenTrack model
        
        This is a wrapper around SirenVis.forward() that takes a batch of
        qpoints, finds their predicted visbilities, and then sums them up
        according to their charge values, resulting in a single predicted
        photo-electron PMT spectrum for the entire track.
        
        Note that the input coordinates are expected to be normalized to
        [-1,1], which is done in the dataloader (via `PhotonLib.meta.norm_coord`).
        
        Parameters
        ----------
        batch : dict
            Batch of tracks. Expected keys are:
            - qpt_v : torch.Tensor of shape (N,4)
                Flattened tensor of qpoints (x,y,z,q) containing many tracks.
            - q_sizes : torch.Tensor of shape (N,)
                The number of qpoints per track so they can be reconstructed.
        """
        
        qpt_v = batch['qpt_v']
        charge_coords = qpt_v[:,:3]
        vis = super().forward(charge_coords)       # vis in log scale

        # inv transform of SIREN output
        vis = self._inv_xform_vis(vis)          # vis in linear scale
        
        charge_size = batch['q_sizes']
        # TODO (ys nov 20 2023): shouldnt be *too* expensive, but can do in dataloader
        toc = np.concatenate([[0], np.cumsum(charge_size.to('cpu'))])
        
        pred = []
        charge_value = qpt_v[:,3]
        for i in range(len(charge_size)):
            start, end = toc[i], toc[i+1]
            sum_q_vis = (vis[start:end]*charge_value[start:end].unsqueeze(-1)).sum(axis=0)
            pred.append(sum_q_vis)

        pe_pred = torch.stack(pred).to(charge_coords.device)        
        
        out = {
            'pe_v':pe_pred,
            'vis':vis,
            }
        return out
