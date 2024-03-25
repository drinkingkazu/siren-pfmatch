from __future__ import annotations

import numpy as np
import torch

from photonlib import PhotonLib, MultiLib
from slar.nets import SirenVis, MultiVis
from pfmatch.algorithms import LightPath
from pfmatch.datatypes import Flash, FlashMatchInput, QCluster

class ToyMC():
    #TODO: Modify to work with photon library or siren input for visibility

    def __init__(self, cfg:dict, detector_specs: dict, plib: PhotonLib | MultiLib | SirenVis | MultiVis):
        
        gen_cfg = cfg['ToyMC']
        self.time_algo  = gen_cfg["TimeAlgo"]
        self.track_algo = gen_cfg["TrackAlgo"]
        self.periodTPC  = gen_cfg["PeriodTPC"]
        self.periodPMT  = gen_cfg["PeriodPMT"]
        self.ly_variation = gen_cfg["LightYieldVariation"]
        self.pe_variation = gen_cfg["PEVariation"]
        self.posx_variation = gen_cfg['PosXVariation']
        self.truncate_tpc = gen_cfg["TruncateTPC"]
        self.neutrino_time = gen_cfg["NeutrinoTime"]
        self.rng = np.random.default_rng(gen_cfg.get('NumpySeed'))

        num_tracks = gen_cfg['NumTracks']
        if isinstance(num_tracks,int) or (isinstance(num_tracks,str) and num_tracks.isdigit()):
            self.num_tracks = [int(num_tracks)]*2
        elif isinstance(num_tracks,str):
            num_range = num_tracks.split('-')
            if len(num_range)==2 and num_range[0].isdigit() and num_range[1].isdigit():
                i0 = int(num_range[0])
                i1 = int(num_range[1])
                assert i0 <= i1, f'Range low end {i0} must be equal or smaller than the high end {i1}'
                self.num_tracks=[i0,i1]
            else:
                raise ValueError('NumTracks argument must be an integero or "N-M" string format')
        else:
            raise ValueError(f'NumTracks argument invalid: {num_tracks}')

        self.detector = detector_specs
        self.plib = plib
        self.qcluster_algo = LightPath(cfg,self.detector)
        
        n_out = self.plib.n_pmts
        if n_out is None:
            raise AttributeError(
                'No method to get n_pmts from', type(self.plib)
            )

        self.pe_var = torch.ones(n_out)
        if self.pe_variation>0.:
            self.pe_var = abs(self.rng.normal(1.0, self.pe_variation, n_out))
        
    def make_flashmatch_inputs(self, num_match=None):
        """
        Make N input pairs for flash matching
        --------
        Arguments
        --------
        Returns
        Generated trajectory, tpc, pmt, and raw tpc arrays
        """
        #num_tracks = 10

        if num_match is None:
            num_match = self.num_tracks[0]
            num_match += int((self.num_tracks[1]-self.num_tracks[0]+1) * self.rng.random())

        result = FlashMatchInput()

        # generate 3D trajectories inside the detector
        track_v = self.gen_trajectories(num_match)
        result.track_v = track_v

        # generate flash time and x shift (for reco x position assuming trigger time)
        xt_v = self.gen_xt_shift(len(track_v))

        # Defined allowed x recording regions in "reconstructed x" coordinate (assuming neutrino timing)
        min_tpcx, max_tpcx = [t * self.detector['DriftVelocity'] for t in self.periodTPC]

        # generate flash and qclusters in 5 steps
        for idx, track in enumerate(track_v):
            
            # 1. create raw TPC position and light info
            qpt_v = self.make_qpoints(track)

            #raw_qcluster.idx = idx
            
            # 2. Create PMT PE spectrum from raw qcluster
            pe_v,pe_err_v,pe_true_v = self.make_photons(qpt_v)
            #flash.idx = idx
            
            # 3. Apply x shift and set flash time
            ftime, dx = xt_v[idx]
            #flash.time = ftime
            #flash.time_true = ftime            
            #qcluster = raw_qcluster.shift(dx)
            #qcluster.idx = idx
            #qcluster.time_true = ftime
            #raw_qcluster.time_true = ftime
            
            raw_qcluster = QCluster(qpt_v, idx, time=self.neutrino_time, xmin_true=qpt_v[:,0].min().item())
            qcluster = raw_qcluster.copy()
            qcluster.shift(dx)
            flash = Flash(pe_v, pe_true_v, pe_err_v, 
                idx=idx, time=ftime, time_true=ftime, time_width=8. )

            # 4. Drop qcluster points that are outside the recording range
            if self.truncate_tpc:
                qcluster.drop(min_tpcx, max_tpcx)
                
            # 5. check for orphan
            valid_q = len(qcluster) > 0
            valid_f = flash.sum() > 0

            if valid_q and valid_f:
                result.true_match.append((len(result.qcluster_v),len(result.flash_v)))

            if valid_q:
                result.qcluster_v.append(qcluster)
                result.raw_qcluster_v.append(raw_qcluster)
                
            if valid_f:
                result.flash_v.append(flash)
        return result

    def gen_trajectories(self, num_tracks):
        """
        Generate N random trajectories.
        ---------
        Arguments
            num_tracks: int, number of tpc trajectories to be generated
        -------
        Returns
            a list of trajectories, each is a pair of 3D start and end points
        """
        #load detector dimension 
        xmin, ymin, zmin = self.detector['ActiveVolumeMin']
        xmax, ymax, zmax = self.detector['ActiveVolumeMax']
        
        if self.track_algo == 'random-extended':
            from pfmatch.utils import (generate_unbounded_tracks,
                                       segment_intersection_point,
                                       is_outside_boundary)

            boundary = ((xmin, xmax), (ymin, ymax), (zmin, zmax))
            tracks = generate_unbounded_tracks(num_tracks, boundary, extension_factor=0.5, rng=self.rng)
            mask = np.asarray([is_outside_boundary(point, boundary) for point in tracks.reshape(-1, 3)]).reshape(-1, 2)
            one_in = mask.sum(axis=1) == 1
            all_in = mask.sum(axis=1) == 0
            tracks_both_in = tracks[all_in]
            tracks_one_in = tracks[one_in]

            for track,m in zip(tracks_one_in, mask[one_in]):
                bad_pt = track[m].squeeze()
                good_pt = track[~m].squeeze()
                
                intersection = segment_intersection_point(boundary, good_pt, bad_pt)[0]
                track[m] = intersection.reshape(-1, 3)
                
            out_tracks = np.vstack([tracks_both_in, tracks_one_in])

            if len(out_tracks) < num_tracks:
                return np.vstack([out_tracks, self.gen_trajectories(num_tracks-len(out_tracks))])
             
            return out_tracks.tolist()

        res = []
        between = lambda min, max: min + (max - min) * self.rng.random()  # noqa: E731
        for _ in range(num_tracks):
            if self.track_algo=="random":
                start_pt = [between(xmin, xmax),
                            between(ymin, ymax),
                            between(zmin, zmax)]
                end_pt = [between(xmin, xmax),
                          between(ymin, ymax),
                          between(zmin, zmax)]
            elif self.track_algo=="top-bottom": 
            #probably dont need
                start_pt = [between(xmin, xmax),
                            ymax,
                            between(zmin, zmax)]
                end_pt = [between(xmin, xmax),
                          ymin,
                          between(zmin, zmax)]
            else:
                raise ValueError("Track algo not recognized, must be one of ['random', 'top-bottom']")
            res.append([start_pt, end_pt])

        return res

    def gen_xt_shift(self, n):
        """
        Generate flash timing and corresponding X shift
        ---------
        Arguments
            n: int, number of track/flash (number of flash time to be generated)
        -------
        Returns
            a list of pairs, (flash time, dx to be applied on TPC points)
        """
        #can be configured with config file, but default in previous code is to not have one
        #time_algo = 'random'
        #periodPMT = [-1000, 1000]

        time_dx_v = []
        duration = self.periodPMT[1] - self.periodPMT[0]
        for idx in range(n):
            t,x=0.,0.
            if self.time_algo == 'random':
                t = self.rng.random() * duration + self.periodPMT[0]
            elif self.time_algo == 'periodic':
                t = (idx + 0.5) * duration / n + self.periodPMT[0]
            elif self.time_algo == 'same':
                t = self.neutrino_time
            else:
                raise ValueError("Time algo not recognized, must be one of ['random', 'periodic']")
            x = (t - self.neutrino_time) * self.detector['DriftVelocity']
            time_dx_v.append((t,x))
        return time_dx_v

    def make_qpoints(self, track):
        """
        Create a qcluster instance from a trajectory
        ---------
        Arguments 
            track: trajectory defined by 3D points
        -------
        Returns
            a qcluster instance 
        """
        #ly_variation = 0.0
        #posx_variation = 0.0

        qpt_v = self.qcluster_algo.track_to_qpoints(track)
        # apply variation if needed
        if self.ly_variation > 0:
            var = abs(self.rng.normal(1.0, self.ly_variation, len(qpt_v)))
            for idx in range(len(qpt_v)):
                qpt_v[idx][-1] *= var[idx]
        if self.posx_variation > 0:
            var = abs(self.rng.normal(0.0, self.posx_variation, len(qpt_v)))
            for idx in range(len(qpt_v)):
                qpt_v[idx][0] += var[idx]

        return qpt_v

    def make_photons(self, qpt_v):
        """
        Create a flash instance from a qcluster
        ---------
        Arguments
            qcluster: array of 3D position + charge
        -------
        Returns
            a flash instance 
        """
        pe_true_v = torch.sum(self.plib.visibility(qpt_v[:,:3])*(qpt_v[:, 3].unsqueeze(-1)), axis = 0).int().float()
        pe_v = pe_true_v.detach().clone()
        pe_err_v = torch.zeros_like(pe_v)
        for idx in range(len(pe_v)):
            estimate = self.rng.poisson(pe_v[idx].item() * self.pe_var[idx])
            pe_v[idx] = estimate
            pe_err_v[idx] = np.sqrt(estimate)

        return pe_v, pe_err_v, pe_true_v

def attribute_names():
    return [
        'idx',
        'flash_idx',
        'qcluster_idx',
        'raw_qcluster_min_x',
        'raw_qcluster_max_x',
        'qcluster_min_x'
        'qcluster_max_x'
        'qcluster_sum',
        'qcluster_len',
        'qcluster_time_true',
        'flash_sum',
        'flash_time_true',
        'flash_dt_prev',
        'flash_dt_next',
    ]
