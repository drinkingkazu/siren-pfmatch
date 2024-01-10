import numpy as np
import torch
import yaml


class LightPath():

    def __init__(self, cfg: dict = dict(), detector_specs: dict = dict()):

        self._gap = cfg.get('LightPath', dict()).get('SegmentSize',0.5) # in cm

        self._dEdxMIP = detector_specs.get('MIPdEdx',2.1)
        self._light_yield = detector_specs.get('LightYield',24000.)
        
    
    @property
    def gap(self):
        return self._gap
    
    @property
    def dEdxMIP(self):
        return self._dEdxMIP
    
    @property
    def light_yield(self):
        return self._light_yield
    
    
    
    def segment_to_qpoints(self, pt1, pt2) -> torch.Tensor:
        """
        Make a qcluster instance based given trajectory points and detector specs
        ---------
        Arguments
          pt1, pt2: 3D trajectory points
          dedx: stopping power
          light yield: light yield
        -------
        Returns
        """
        norm_alg = torch.linalg.norm
        if not isinstance(pt1, torch.Tensor):
            pt1 = torch.as_tensor(pt1, dtype=torch.float32)
            pt2 = torch.as_tensor(pt2, dtype=torch.float32)


        dist = norm_alg(pt2 - pt1)
        num_div = int(dist / self.gap)
        
        # segment less than gap threshold
        if num_div < 1:
            qpt_v=torch.zeros(size=(1,pt1.shape[0]+1),dtype=pt1.dtype,device=pt1.device)
            qpt_v[0,:-1] = (pt1 + pt2) / 2 
            qpt_v[0, -1] = self.light_yield * self.dEdxMIP * dist
            return qpt_v

        # segment larger than gap threshold
        direct = (pt2 - pt1) / dist
        weight = (dist - int(dist / self.gap) * self.gap)        
        num_pts = num_div if torch.allclose(weight.float(),torch.as_tensor(0.)) else num_div+1

        qpt_v = torch.zeros(size=(num_pts, pt1.shape[0]+1), dtype=pt1.dtype, device=pt1.device)
        offsets = torch.arange(num_pts, dtype=pt1.dtype, device=pt1.device).unsqueeze(1) * self.gap
        offsets += self.gap / 2.0  # for the case of div_idx < num_div
        q = torch.full((num_pts,), self.light_yield * self.dEdxMIP * self.gap, dtype=pt1.dtype, device=pt1.device)
        if num_pts > num_div:
            offsets[num_div:] += (weight - self.gap) / 2.0
            q[num_div:] = self.light_yield * self.dEdxMIP * weight
        qpt_v[:, :-1] = pt1 + direct * offsets
        qpt_v[:, -1] = q
        return qpt_v
        
    def track_to_qpoints(self, track: torch.Tensor) -> torch.Tensor:
        """
        Create a qcluster instance from a trajectory
        ---------
        Arguments
          track: array of trajectory points
        -------
        Returns
          a qcluster instance
        """
        track = torch.as_tensor(track, dtype=torch.float32)

        assert track.dim() == 2, 'The track must be two dimension'
        assert track.shape[0] > 1, 'The track must have >= two points'
                
        qpt_vv=[]

        # add first point of trajectory
        #qpt_v = torch.tensor([track[0][0], track[0][1], track[0][2], 0.]).reshape(1,-1)

        for i in range(len(track)-1):
            qpt_vv.append(self.segment_to_qpoints(track[i],track[i+1]))
            #print(track[i],'=>',track[i+1])
            #print(qpt_vv[-1].shape,'...',qpt_vv[-1][0],'=>',qpt_vv[-1][-1],'\n')
            #self.fill_qcluster(np.array(track[i]), np.array(track[i+1]), res)

        # add last point of trajectory
        #res.qpt_v = torch.concat([res.qpt_v, torch.tensor([track[-1][0], track[-1][1], track[-1][2], 0.], 
        #                                                  device=res.qpt_v.device,
        #                                                 ).reshape(1,-1)])
        
        return torch.cat(qpt_vv)
