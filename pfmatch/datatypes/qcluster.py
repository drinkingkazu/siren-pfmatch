from __future__ import annotations

import torch
import numpy as np
import copy
from photonlib import PhotonLib
from slar.nets import SirenVis

class QCluster:
    def __init__(self, qpt_v, idx=2**31-1, time=np.inf, xmin_true=np.inf):
        self._qpt_v = torch.as_tensor(qpt_v, dtype=torch.float32) # vector of 3D points along track, along with photons "q" originating from each position
        if not (len(self._qpt_v.shape) == 2 and self._qpt_v.shape[1] == 4):
            raise ValueError(f'qpt_v must have shape (N,4), got {self._qpt_v.shape}')

        self._idx = idx # index from original larlite vector
        self._time = time # assumed time w.r.t trigger for reconstruction
        self._xmin_true = xmin_true # time from MCTrack information

    def __len__(self):
        return len(self.qpt_v)

    def __add__(self,other):
        return QCluster(torch.cat([self.qpt_v, other.qpt_v], 0))
    
    def to(self,device):
        self._qpt_v = self._qpt_v.to(device)
        return self

    def min(self,axis):
        return self.qpt_v[:,axis].min()

    @property
    def qpt_v(self):
        return self._qpt_v
    
    @qpt_v.setter
    def qpt_v(self, qpt_v):
        self._qpt_v = qpt_v

    @property
    def idx(self):
        return self._idx

    @property
    def time(self):
        return self._time

    @property
    def xmin_true(self):
        return self._xmin_true

    def copy(self):
        return copy.deepcopy(self)

    # total length of the track
    def length(self):
        res = 0
        for i in range(1, len(self.qpt_v)):
            res += torch.linalg.norm(self.qpt_v[i, :3] - self.qpt_v[i-1, :3]).item()
        return res

    # sum over charge 
    def sum(self):
        if len(self.qpt_v) == 0:
            return 0
        return torch.sum(self.qpt_v[:, -1]).item()

    # sum over x coordinates of the track
    def xsum(self):
        if len(self.qpt_v) == 0:
            return 0
        return torch.sum(self.qpt_v[:, 0]).item()

    # shift qcluster_v by given dx
    def shift(self, dx):
        self.qpt_v[:,0] += dx
        #other = copy.deepcopy(self)
        #other.qpt_v[:, 0] += dx
        #return other

    # drop points outside specified recording range
    def drop(self, x_min, x_max, y_min = -np.inf, y_max = np.inf, z_min = -np.inf, z_max = np.inf):
        mask = (self.qpt_v[:, 0] >= x_min) & (self.qpt_v[:, 0] <= x_max) & \
            (self.qpt_v[:, 1] >= y_min) & (self.qpt_v[:, 1] <= y_max) & \
            (self.qpt_v[:, 2] >= z_min) & (self.qpt_v[:, 2] <= z_max)
        self.qpt_v = self.qpt_v[mask]


    def shift_to_center(self, plib: PhotonLib | SirenVis):
        
        vol_xmin = plib.meta.ranges[0,0]
        vol_xmax = plib.meta.ranges[0,1]
        
        track_xmin = self.qpt_v[:,0].min().item()
        track_xmax = self.qpt_v[:,0].max().item()
        track_span = track_xmax - track_xmin
        
        vol_span = vol_xmax - vol_xmin
        assert track_span < vol_span, f'Track span {track_span} exceeds the volume span {vol_span}'
        
        # how much to shift so that the center of the track is at the center of the volume?
        track_xmid = track_span/2. + track_xmin
        vol_xmid = vol_span/2. + vol_xmin
        xshift = vol_xmid - track_xmid
        self.shift(xshift)
                
    def shift_to_xmin(self, plib: PhotonLib | SirenVis):
        
        vol_xmin = plib.meta.ranges[0,0]
        track_xmin = self.qpt_v[:,0].min().item()
        xshift = vol_xmin - track_xmin
        self.shift(xshift)
        
    def shift_to_xmax(self, plib: PhotonLib | SirenVis):
        
        vol_xmax = plib.meta.ranges[0,1]
        track_xmax = self.qpt_v[:,0].max().item()
        xshift = vol_xmax - track_xmax
        self.shift(xshift)