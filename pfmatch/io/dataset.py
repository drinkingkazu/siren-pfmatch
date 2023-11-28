import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from pfmatch.utils import load_detector_config


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

        self._cfg = cfg.get('data')        
        fname = self._cfg['dataset']['filepath']

        self._plib = None
        self._toymc = None

        self.qpt_vv = None
        self.qpt_vv_sizes = None
        self.qpt_vv_toc = None
        self.pe_vv = None
        self.n_tracks = self._cfg['dataset'].get('size')   
        if os.path.exists(fname):
            self.load(fname)
            return
        
        from photonlib import PhotonLib

        from pfmatch.apps import ToyMC

        print('[ToyMCDataset] dataset file',fname,'not found... generating via ToyMC')
        self._plib = PhotonLib.load(cfg)
        
        self._toymc = ToyMC(cfg, load_detector_config(cfg), self._plib)
        if self.n_tracks is None:
            raise RuntimeError(f'[ToyMCDataset] dataset file {fname} not found and size not specified')

        self.generate(self.n_tracks, fname, batch_size=self._cfg['dataset'].get('batch_size'))
        self.load(fname)
        
    @property
    def toymc(self):
        return self._toymc
    
    @property
    def plib(self):
        return self._plib
        
    def __len__(self):
        """number of tracks in the dataset"""
        if self.n_tracks:
            return self.n_tracks
        
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
            
        # set the loss weighting factor matrix
        self.weights = np.ones_like(self.pe_vv)
        weight_cfg = self._cfg['dataset'].get('weight')
        if weight_cfg:
            print('[ToyMCDataset] weighting the loss using',weight_cfg.get('method'))
            print('[ToyMCDataset] params:', weight_cfg)
            if weight_cfg.get('method') == 'pe':
                self.weights = self.pe_vv * weight_cfg.get('factor', 1.)
            else:
                raise NotImplementedError(f'The weight mode {weight_cfg.get("method")} is invalid.')
            
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
        pbar = tqdm(total=n_tracks, desc='Generating ToyMCDataset', unit='track')
        while curr < len(tracks):

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

            pbar.update(batch_size)

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
        weights = self.weights[idx]

        output = {
            'qpt_v':qpt_vv.tolist(),
            'pe_v':pe_v.tolist(),
            'q_sizes':q_sizes.tolist(),
            'weights': weights.tolist(),
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
        output['charge_csum'] = torch.concatenate([torch.tensor([0], dtype=torch.int32), torch.cumsum(output['q_sizes'], dim=0)], dim=0)
        output['weights'] = torch.as_tensor([data['weights'] for data in batch], dtype=torch.float32)
        return output
