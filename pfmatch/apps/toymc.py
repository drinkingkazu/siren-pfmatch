from __future__ import annotations

import numpy as np
import torch
import os
import h5py
from torch.utils.data import Dataset
from tqdm import tqdm

from photonlib import PhotonLib
from slar.nets import SirenVis
from pfmatch.algorithms import LightPath
from pfmatch.utils import load_detector_config
from pfmatch.datatypes import Flash, FlashMatchInput, QCluster

class ToyMC():
    #TODO: Modify to work with photon library or siren input for visibility

    def __init__(self, cfg:dict, detector_specs: dict, plib: PhotonLib | SirenVis):
        
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
                raise ValueError(f'NumTracks argument must be an integero or "N-M" string format')
        else:
            raise ValueError(f'NumTracks argument invalid: {num_tracks}')

        self.detector = detector_specs
        self.plib = plib
        self.qcluster_algo = LightPath(cfg,self.detector)
        
        n_out = self.plib._n_outs if hasattr (self.plib, '_n_outs') else len(self.plib)
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
            valid_match = len(qcluster) > 0 and flash.sum() > 0
            
            if len(qcluster) > 0:
                result.qcluster_v.append(qcluster)
                result.raw_qcluster_v.append(raw_qcluster)
                
            if flash.sum() > 0:
                result.flash_v.append(flash)
                
            if valid_match:
                result.true_match.append((idx,idx))

        return result

    def gen_trajectories(self, num_tracks, seed=123):
        """
        Generate N random trajectories.
        ---------
        Arguments
            num_tracks: int, number of tpc trajectories to be generated
        -------
        Returns
            a list of trajectories, each is a pair of 3D start and end points
        """
        #track_algo = 'random'

        res = []

        #load detector dimension 
        xmin, ymin, zmin = self.detector['ActiveVolumeMin']
        xmax, ymax, zmax = self.detector['ActiveVolumeMax']
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
        xsum = 0. if len(qpt_v)<1 else torch.sum(qpt_v[:, 0]).item()
        # apply variation if needed
        if self.ly_variation > 0:
            var = abs(self.rng.normal(1.0, self.ly_variation, len(qpt_v)))
            for idx in range(len(qpt_v)):
                qpt_v[idx][-1] *= var[idx]
        if self.posx_variation > 0:
            var = abs(self.rng.normal(1.0, self.posx_variation/xsum, len(qpt_v)))
            for idx in range(len(qpt_v)):
                qpt_v[idx][0] *= var[idx]

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
    
class ToyMCDataset(Dataset):
    """
    ToyMC track generation in the form of torch Dataset for training SirenVis
    """
    
    def __init__(self, cfg: dict):
        """Creates a torch Dataset object for training SirenVis with LArTPC images
        containing one track.
        
        A h5 file is specified in the configuration file. It must be of the following
        format:
        - qpt_v_normed (N_points, 4): normalized (i.e. on range[-1,1]) 
          coordinates and charge (x,y,z,q).
        - num_qpt_v (N_track): number of points in each track, so we can reconstruct each
          track individually.
        - pe_v (N_track, N_pmt): photo-electron PMT spectrum.
        
        If the file exists, it is loaded. If not, ToyMC will be used to generate the
        dataset according to specifications in the config, and then load it.
        
        Configuration parameters:
        -------------------------
        no matter what:
        - `data.dataset.filename`: h5 file to load or save the dataset
        if the h5 file does not exist:
        - `data.dataset.size`: number of tracks to generate if the h5 file does not exist
        - `data.dataset.batch_size`: batch size for the dataloader
        - all ToyMC parameters
        - all PhotonLib parameters

        Parameters
        ----------
        cfg : dict
            model configuration.
        plib : PhotonLib
            photon library to use for input to the ToyMC object
        """

        this_cfg = cfg.get('data')        
        fname = this_cfg['dataset']['filepath']

        self._plib = None
        self._toymc = None

        self.qpt_vv = None
        self.qpt_vv_sizes = None
        self.qpt_vv_toc = None
        self.pe_vv = None
                
        if os.path.exists(fname):
            self.load(fname)
            return
        
        from photonlib import PhotonLib
        from pfmatch.apps import ToyMC

        print('[ToyMCDataset] dataset file',fname,'not found... generating via ToyMC')
        self._plib = PhotonLib.load(cfg)
        self._toymc = ToyMC(cfg, load_detector_config(cfg), self._plib)
        n_tracks = this_cfg['dataset'].get('size')
        if n_tracks is None:
            raise RuntimeError(f'[ToyMCDataset] dataset file {fname} not found and size not specified')

        self.generate(n_tracks, fname, batch_size=this_cfg['dataset'].get('batch_size'))
        self.load(fname)
        
        # todo: vis weights like in PhotonLibDataset?
        
    @property
    def toymc(self):
        return self._toymc
    
    @property
    def plib(self):
        return self._plib
        
    def __len__(self):
        """number of tracks in the dataset"""
        return len(self.qpt_vv_sizes)
    
    def load(self, fname: str):
        """Loads in a dataset of ToyMC tracks from the specified h5 file.
        
        The h5 file should contain the following datasets:
        - qpt_v_normed (N_points, 4): normalized (i.e. on range[-1,1]) coordinates and charge (x,y,z,q).
        - num_qpt_v (N_track): number of points in each track, so we can reconstruct each track individually.
        - pe_v (N_track, N_pmt): photo-electron PMT spectrum. N_pmt is set in PhotonLib.
        
        Note that qpt_v_normed is a tensor or many tracks concatenated together, so we need num_qpt_v to
        reconstruct them. (That is to say sum(num_qpt_v) = N_points.)

        Parameters
        ----------
        fname : str
            Filename to load the dataset from.
        """

        print('[ToyMCDataset] loading',fname)
        with h5py.File(fname,'r') as f:
            self.qpt_vv = f['qpt_v_normed'][:]
            self.qpt_vv_sizes = f['num_qpt_v'][:]
            self.pe_vv = f['pe_v'][:]
        self.qpt_vv_toc = np.concatenate([[0], np.cumsum(self.qpt_vv_sizes)])
    
    def generate(self, n_tracks: int, fname: str, batch_size: int = None):
        """Generates ToyMC tracks and saves it to the specified h5 file.
    
        n_tracks can be arbitrarily large, but the dataset is generated in batches of batch_size
        to avoid memory issues.
        
        Refer to `ToyMCDataset.load` for the format of the h5 file.

        Parameters
        ----------
        fname : str
            Filename to load the dataset from.
        """
        print('[ToyMCDataset] generating',n_tracks,'tracks in',fname)
        if batch_size is None:
            batch_size = n_tracks // 10 if n_tracks > 100 else 1

        # create h5 file
        dataset = h5py.File(fname, 'w')
        dataset.create_dataset('qpt_v_normed', shape=(0, 4), maxshape=(None, 4), dtype='float32')
        dataset.create_dataset('num_qpt_v', shape=(0,), maxshape=(None,), dtype='int16')
        dataset.create_dataset('pe_v', shape=(0, self.plib.n_pmts), maxshape=(None, self.plib.n_pmts), dtype='float32')
        dataset.close()
        
        # generate ToyMC tracks
        tracks = self.toymc.gen_trajectories(n_tracks)
        
        curr = 0
        pbar = tqdm(total=n_tracks//batch_size, desc='Generating ToyMCDataset', unit='batch')
        while curr < len(tracks):
            pbar.update(1)

            dataset = h5py.File(fname, 'a')

            # make qpoints and photoelectron spectra based on those qpoints in batches
            qpt_vv = list(map(self.toymc.make_qpoints, tracks[curr:curr+batch_size]))
            pe_vv = np.asarray(list(map(lambda x: self.toymc.make_photons(x)[0].detach().cpu().tolist(), qpt_vv)))
            curr += batch_size
            
            for qpt_v in qpt_vv:
                # normalize coordinates to [-1,1]
                qpt_v[:,:3] = self.plib.meta.norm_coord(qpt_v[:,:3])
                    
                # save to h5 file
                dataset['qpt_v_normed'].resize((dataset['qpt_v_normed'].shape[0] + qpt_v.shape[0]), axis=0)
                dataset['qpt_v_normed'][-qpt_v.shape[0]:] = qpt_v
                dataset['num_qpt_v'].resize((dataset['num_qpt_v'].shape[0] + 1), axis=0)
                dataset['num_qpt_v'][-1] = qpt_v.shape[0]

            # pe_v is always the same shape so we don't need to add em individually
            dataset['pe_v'].resize((dataset['pe_v'].shape[0] + pe_vv.shape[0]), axis=0)
            dataset['pe_v'][-pe_vv.shape[0]:] = pe_vv

            dataset.close()

    def __getitem__(self, idx: int) -> dict:
        """Used by torch Dataset to get a single track.
        
        Parameters
        ----------
        idx : int
            index of the track

        Returns
        -------
        dict
            a dictionary containing the track's qpt_v, pe_v, and q_sizes.
        """

        start, end = self.qpt_vv_toc[idx], self.qpt_vv_toc[idx+1]
        qpt_vv = self.qpt_vv[start:end]
        pe_v = self.pe_vv[idx]
        q_sizes = self.qpt_vv_sizes[idx]

        output = {
            'qpt_v':qpt_vv.tolist(),
            'pe_v':pe_v.tolist(),
            'q_sizes':q_sizes.tolist(),
            # todo: weight?
        }
        
        return output
    
    @staticmethod
    def collate_fn(batch: dict) -> dict:
        """
        Used by torch DataLoader to collate a batch of tracks together into a single dictionary.
        """
        cat = lambda x : np.squeeze(x) if len(batch) <= 1 else np.concatenate(x) # noqa: E731
        output = {}
        output['qpt_v'] = torch.as_tensor(
                    cat([data['qpt_v'] for data in batch]), dtype=torch.float32
            )
        output['pe_v'] = torch.as_tensor([data['pe_v'] for data in batch], dtype=torch.float32)
        output['q_sizes'] = torch.as_tensor([data['q_sizes'] for data in batch], dtype=torch.int32)
        return output
