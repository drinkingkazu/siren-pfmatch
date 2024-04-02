import os
import torch

from torch.utils.data import Dataset, RandomSampler, BatchSampler, DataLoader
from contextlib import closing
from pfmatch.io import H5File

class SirenCalibDataset(Dataset):
    def __init__(self, filepath, n_pmts):
        super().__init__()
        
        if isinstance(filepath, str):
            filepath = [filepath]
        
        files = filepath
        self._build_file_index(files)
        self._n_pmts = n_pmts
    
    def _build_file_index(self, files):
        cnts = torch.zeros(len(files), dtype=int)
        
        self._files = []
        for i, fpath in enumerate(files):
            if not os.path.isfile(fpath):
                raise FileNotFoundError(fpath)
                
            #with closing(H5File.open(fpath, 'r', verbose=False)) as f:
            #    cnts[i] = len(f)
            f = H5File.open(fpath, 'r', verbose=False)
            self._files.append(f)
            cnts[i] = len(f)
        
        #self._files = files
        self._file_idxs = torch.cat([torch.tensor([0]), cnts.cumsum(0)])
                   
    def __del__(self):
        while len(self._files) > 0:
            f = self._files.pop()
            f.close()

    def __len__(self):
        return self._file_idxs[-1]

    def __getitem__(self, idx):
        return idx

    def read_many(self, idx):
        idx = torch.as_tensor(idx)
        file_idxs = torch.bucketize(idx, self._file_idxs, right=True) - 1
        
        key_q = 'qpt_v'
        key_f = 'pe_v'
        
        output = {
            'qclusters' : [],
            'flashes' : [],
            'ev_idxs' : [],
            'file_idxs' : [],
        }
        
        for fid in file_idxs.unique():
            mask = file_idxs==fid
            offset = self._file_idxs[fid]
            ev_idxs = (idx[mask] - offset).tolist()
            ev_idxs.sort()
        
            #fpath = self._files[fid]
            #with closing(H5File.open(fpath, 'r', verbose=False)) as fin:
            #    q_vv, f_vv = fin.read_many(ev_idxs, self._n_pmts)
            #del fin
            fin = self._files[fid]
            q_vv, f_vv = fin.read_many(ev_idxs, self._n_pmts)
               
            # fill qclusters
            for q_v in q_vv:
                # correct for the true position
                qpts = getattr(q_v[0], key_q)
                qpts[:,0] += q_v[0].xmin_true - qpts[:,0].min()
                output['qclusters'].append(qpts)
                #output['qclusters'].append(getattr(q_v[0], key_q))
            
            # fill flashes
            for f_v in f_vv:
                output['flashes'].append(getattr(f_v[0], key_f))
         
            # fill file and event indices
            output['ev_idxs'].append(ev_idxs)
            output['file_idxs'].append(fid)
            
        return output

    @classmethod
    def factory(cls, cfg):
        ds_cfg = cfg['data']['dataset'].copy()

        filepath = ds_cfg.pop('filepath')

        if isinstance(filepath, str):
            filepath = [filepath]

        from glob import glob
        files = []
        for fpath in filepath:
            files.extend(glob(fpath))

        ds_cfg['filepath'] = files
        return cls(**ds_cfg)

def loader_factory(cfg):
    ds = SirenCalibDataset.factory(cfg)

    data_cfg = cfg['data']

    loader_cfg = data_cfg.get('loader', {})
    loader = DataLoader(
        ds, collate_fn=ds.read_many, **loader_cfg
    )
    return loader
