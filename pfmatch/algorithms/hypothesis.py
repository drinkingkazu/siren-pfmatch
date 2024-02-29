from __future__ import annotations
from itertools import compress

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
        grad_input = torch.matmul(grad_plib, grad_output.unsqueeze(-1))
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

class PLibPrediction(torch.autograd.Function):
    '''
    Autograd function for PhotonLib. Apply for batch input of multiple charge
    clusters.
    '''
    
    @staticmethod
    def forward(ctx, dx, batch, sizes, plib):
        coords = batch[:,:3].clone()
        coords[:,0] += dx.repeat_interleave(sizes)
    
        q = batch[:,3]
        
        vox_ids = plib.meta.coord_to_voxel(coords)
        sizes = list(sizes.cpu())
        
        ctx.save_for_backward(vox_ids, q)
        ctx.plib = plib
        ctx.sizes = sizes
        
        split_vis_q = torch.split(plib.vis[vox_ids]*q.unsqueeze(-1), sizes)
        
        pred = torch.stack([vis_q.sum(axis=0) for vis_q in split_vis_q])
        return pred
            
    @staticmethod
    def backward(ctx, grad_output):
        vox_ids, q = ctx.saved_tensors
        plib = ctx.plib
        sizes = ctx.sizes
        
        grad_pred_x = (plib.vis[vox_ids+1] - plib.vis[vox_ids])
        grad_pred_x /= plib.meta.voxel_size[0]
        grad_pred_x *= q.unsqueeze(-1)
        
        grad_pairs = zip(
            torch.split(grad_pred_x, sizes),
            grad_output,
        )
        
        grad_x = torch.cat([
            grad_in.sum(axis=0).matmul(grad_out).unsqueeze(-1)
            for grad_in, grad_out in grad_pairs
        ])
            
        return grad_x, None, None, None

class MultiFlashHypothesis(torch.nn.Module):
    '''
    Flash hypothesis for batch input that contains multiple charge clusters.
    Each charge cluster has its own optimizible x-offset.

    Arguments
    ---------
    vis_model : PhotonLib | SirenVis
        Visibilty model in form of PhotonLib (LUT) or SirenVis.

    dx_ranges: array_like (N, 2)
        List of x-offset ranges for _N_ flash hypothesis.

    dx_init: array_like (N,), optional
        Initial x-offset for _N_ flashes. Default: 0.
        `len(dx_init) == len(dx_ranges)`
    '''
    def __init__(self, 
                 vis_model : PhotonLib | SirenVis, 
                 dx_ranges, 
                 dx_init=None,
                 ):
        super().__init__()
        self.vis_model = vis_model
        self._dx_ranges = torch.as_tensor(dx_ranges)

        if dx_init is None:
            dx_init = torch.zeros(len(self))

        dx_init = torch.as_tensor(dx_init)
        if len(dx_init) != len(self):
            raise ValueError(f'len(dx_init) != len(dx_ranges)')

        self.pars = torch.nn.ParameterList([
            torch.nn.Parameter(dx0) for dx0 in dx_init
        ])

    def __len__(self):
        return len(self._dx_ranges)
    
    def mask_dx(self, mask=None, clamp=False):
        '''
        Offset parameters with optional mask.

        Arguments
        ---------
        mask: Tensor(bool), optional
            Mask to be applied. Default: None.
            If `None`, returns all parameters.

        clamp: bool, optional
            Clamp offset parameters within ``_dx_ranges``.
            Default: `False`

        Returns
        -------
        dx: Tensor, (_N_,)
            Offset parameters. If masked, `N <= len(self)`.
            Otherwise, `N == len(self)`.
        '''
        dx_ranges = self._dx_ranges
        if mask is None:
            pars = torch.stack(list(self.pars))
        else:
            dx_ranges = dx_ranges[mask]
            pars = torch.stack(list(compress(self.pars, mask)))
            
        if clamp:
            pars = pars.clamp(*dx_ranges.T)
            
        return pars
               
    @property
    def dx(self):
        '''
        Offset parameters without mask and clamp.
        '''
        return self.mask_dx()
    
    @dx.setter
    def dx(self, values):
        if len(self.pars) != len(values):
            raise ValueError(f'len(values) != len(pars)')

        for p, v in zip(self.pars, values):
            p.data.fill_(v)

    def forward(self, batch, sizes, mask=None):
        '''
        Predict optical output for batch input of charge clusters.

        Arguments
        ---------
        batch: Tensor
            A single tensor of all charge cluster points. May contains multiple
            clusters.

        sizes: Tensor
            Number of points for each charge clusters.
            `len(sizes)` represents the number of clusters in `batch`.
            `sum(sizes) == len(batch)` gives the total number of charge points.

        mask: Tensor(bool), optional
            Mask to be applied. Default `None`.
            If mask is not set, all clusters in `batch` are evalulated.
            Otherwise, `len(mask) == len(sizes)`, only the masked clusters are
            included.

        Returns
        -------
        output: Tensor, (_N_,_M_)
            Optical output prediction with the current x-offsets.
            If masked, `N` is the number of masked clusters. Otherwise `N ==
            len(self)` for all clusters.
            `M` is the number of the PMTs.
        '''

        dx = self.mask_dx(mask, clamp=True)
        
        if mask is not None:
            batch = batch[mask.repeat_interleave(sizes)]
            sizes = sizes[mask]
            
        plib = self.vis_model
        output = PLibPrediction.apply(dx, batch, sizes, plib)

        return output
