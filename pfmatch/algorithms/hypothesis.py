from __future__ import annotations

import torch
from photonlib import PhotonLib
from slar.nets import SirenVis


class F(torch.autograd.Function):
    """
        Custom autograd function for PhotonLib
    """

    @staticmethod
    def forward(ctx, track, plib):
        vox_ids = plib.meta.coord_to_voxel(track[:,:3])
        ctx.save_for_backward(track)
        ctx.plib = plib
        pe_v = torch.sum(plib.vis[vox_ids]*(track[:, 3].unsqueeze(-1)), axis = 0)        
        return pe_v

    @staticmethod
    def backward(ctx, grad_output):
        track = ctx.saved_tensors[0]
        vox_ids = ctx.plib.meta.coord_to_voxel(track[:,:3])
        
        grad_plib = (ctx.plib.vis[vox_ids+1] - ctx.plib.vis[vox_ids]) / ctx.plib.meta.voxel_size[0]
        grad_input = torch.matmul(grad_plib*track[:,3].unsqueeze(-1), grad_output.unsqueeze(-1))
        pad = torch.zeros(grad_input.shape[0], 3, device=grad_output.device)
        return torch.cat((grad_input,pad), -1), None

class FlashHypothesis(torch.nn.Module):

    def __init__(self, cfg:dict, plib:PhotonLib | SirenVis): 
        super().__init__()
        self._plib = plib
        # fmatch_cfg = cfg.get('FlashHypothesis',dict())

        self._dx = torch.nn.Parameter(torch.as_tensor([0.]))
        self._dx.data.fill_(0.)

        self._xmin = self.plib.meta.ranges[0,0].item()
        self._xmax = self.plib.meta.ranges[0,1].item()

        self._dx_min = self._xmin
        self._dx_max = self._xmax

    @property
    def plib(self):
        return self._plib

    @property
    def dx(self):
        return self._dx.item()

    @dx.setter
    def dx(self, value):
        return self._dx.data.fill_(value)

    def dx_range(self, track):
        track_xmin = torch.min(track[:,0]).item()
        track_xmax = torch.max(track[:,0]).item()
        assert self._xmin <= track_xmin, f'Track minimum x {track_xmin} is outside the volume minimum x {self._xmin}'
        assert self._xmax >= track_xmax, f'Track maximum x {track_xmax} is outside the volume maximum x {self._xmax}'

        dx_min = self._xmin - track_xmin
        dx_max = self._xmax - track_xmax
        return dx_min, dx_max
        
    def forward(self,track: torch.Tensor):
        """
        fill flash hypothsis based on given qcluster track
        ---------
        Arguments
        track: qcluster track of 3D position + charge
        ---------
        Returns hypothesized number of p.e. to be detected in each PMT
        """

        self._dx_min, self._dx_max = self.dx_range(track)
        self._dx.data.clamp_(self._dx_min, self._dx_max)
        shift = torch.cat((self._dx, torch.zeros(3, device=track.device)), -1)
        shifted_track = torch.add(track, shift.expand(track.shape[0], -1))
        if isinstance(self._plib,PhotonLib):
            return F.apply(shifted_track,self.plib)
        else:
            x = self.plib.meta.norm_coord(shifted_track[:,:3])
            pe_v = torch.sum(self.plib._inv_xform_vis(self.plib(x))*(shifted_track[:, 3].unsqueeze(-1)), axis = 0)
            return pe_v

