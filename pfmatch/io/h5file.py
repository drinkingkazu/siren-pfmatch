from typing import List
import h5py as h5
import numpy as np
import torch

from pfmatch.datatypes import Flash, QCluster

class H5File(object):
    '''
    Class to interface with the HDF5 dataset for pfmatch. 
    It can store arbitrary number of events.
    Each event contains arbitrary number of QCluster and Flash using a "ragged array".
    '''
    
    def __init__(self,cfg:dict):
        '''
        Constructor
            file_name: string path to the data file to be read/written
            mode: h5py.File constructor "mode" (w=write, a=append, r=read)
        '''
        file_cfg = cfg.get('H5File',dict())
        file_name = file_cfg.get('FileName','garbage.h5')
        mode = file_cfg.get('Mode','w')
        self._open(file_name,mode)

    @classmethod
    def open(cls,file_name:str,mode:str): 
        cfg = dict(H5File=dict(FileName=file_name,Mode=mode))
        return cls(cfg)

    def _open(self, file_name:str, mode:str):

        self._file_name = file_name
        self._mode = mode
        dt_float = h5.vlen_dtype(np.dtype('float32'))
        dt_int   = h5.vlen_dtype(np.dtype('int32'))
        print(f'[H5File] opening {file_name} in mode {mode}')
        self._f = h5.File(self._file_name,self._mode)
        if self._mode in ['w','a']:
            self._wh_point = self._f.create_dataset('point', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_group = self._f.create_dataset('group', shape=(0,), maxshape=(None,), dtype=dt_int  )
            self._wh_flash = self._f.create_dataset('flash', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_match = self._f.create_dataset('match', shape=(0,), maxshape=(None,), dtype=dt_int  )
            self._wh_flash_true = self._f.create_dataset('flash_true', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_flash_err  = self._f.create_dataset('flash_err',  shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_ftrue_group = self._f.create_dataset('ftrue_group', shape=(0,), maxshape=(None,), dtype=dt_int)
            self._wh_ferr_group  = self._f.create_dataset('ferr_group',  shape=(0,), maxshape=(None,), dtype=dt_int)
            self._wh_point.attrs['note'] = '"point": 3D points with photon count. Reshape to (N,4) for total of N points. See "group" attributes to split into cluster.'
            self._wh_group.attrs['note'] = '"group": An array of integers = number of points per cluster. The sum of the array == N points of "point" data.'
            self._wh_flash.attrs['note'] = '"flash": Flashes (photo-electrons-per-pmt). Reshape to (K,180) for K flashes' 
            
            self._wh_qidx  = self._f.create_dataset('qidx', shape=(0,), maxshape=(None,), dtype=dt_int)
            self._wh_qtime = self._f.create_dataset('qtime', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_qxmin_true = self._f.create_dataset('qxmin_true', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_fidx  = self._f.create_dataset('fidx', shape=(0,), maxshape=(None,), dtype=dt_int)
            self._wh_ftime = self._f.create_dataset('ftime', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_ftime_true = self._f.create_dataset('ftime_true', shape=(0,), maxshape=(None,), dtype=dt_float)
            self._wh_ftime_width = self._f.create_dataset('ftime_width', shape=(0,), maxshape=(None,), dtype=dt_float)

    @property
    def file_name(self):
        return self._file_name
    
    @property
    def mode(self):
        return self._mode

    @property
    def f(self):
        return self._f

    def close(self):
        print(f'[H5File] closing {self._file_name}')
        self._f.close()
        
    def __del__(self):
        try:
            self._f.close()
        except AttributeError:
            pass
        
    def __len__(self):
        return len(self._f['point'])
    
    def __getitem__(self,idx):
        return self.read_one(idx)
    
    def __str__(self):
        msg=f'{len(self)} entries in this file. Raw hdf5 attribute descriptions below.\n'
        for k in self._f.keys():
            try:
                msg += ' '*2 + self._f[k].attrs['note'] + '\n'
            except KeyError:
                pass
        return msg
        
    def read_one(self,idx,n_pmts):
        '''
        Read one event specified by the integer index
        '''
        qcluster_vv,flash_vv,match_vv = self.read_many([idx],n_pmts)
        return (qcluster_vv[0],flash_vv[0],match_vv[0])
            
    def read_many(self,idx_v,n_pmts):
        '''
        Read many event specified by an array of integer indexes
        '''
        flash_vv = []
        qcluster_vv = []
        match_vv = []
        
        for idx in idx_v:
            if idx >= len(self):
                raise IndexError(f'index {idx} out of range (max={len(self)-1})')

        event_point_v = [np.array(data).reshape(-1,4) for data in self._f['point'][idx_v]]
        event_group_v = self._f['group'][idx_v]
        event_flash_v = [np.array(data) for data in self._f['flash'][idx_v]]

        event_flash_err_v  = [np.array(data) for data in self._f['flash_err'][idx_v]]
        event_flash_true_v = [np.array(data) for data in self._f['flash_true'][idx_v]]
        event_ferr_group   = [np.array(data) for data in self._f['ftrue_group'][idx_v]]
        event_ftrue_group  = [np.array(data) for data in self._f['ferr_group'][idx_v]]

        event_qidx_v  = [np.array(data) for data in self._f['qidx'][idx_v]]
        event_qtime_v = [np.array(data) for data in self._f['qtime'][idx_v]]
        event_qxmin_true_v = [np.array(data) for data in self._f['qxmin_true'][idx_v]]
        event_fidx_v  = [np.array(data) for data in self._f['fidx'][idx_v]]
        event_ftime_v = [np.array(data) for data in self._f['ftime'][idx_v]]
        event_ftime_true_v = [np.array(data) for data in self._f['ftime_true'][idx_v]]
        event_ftime_width_v = [np.array(data) for data in self._f['ftime_width'][idx_v]]

        if 'match' in self._f.keys():
            event_match_v = [np.array(data) for data in self._f['match'][idx_v]]
        else:
            event_match_v = []
            for i in range(len(idx_v)):
                paired_idx_v = [[v,v] for v in event_fidx_v[i] if v in event_qidx_v[i]]
                event_match_v.append(paired_idx_v)
            event_match_v = np.array(event_match_v)
        
        for i in range(len(idx_v)):
            
            event_point = event_point_v[i]
            event_group = event_group_v[i]
            event_flash = event_flash_v[i].reshape(-1,n_pmts)
            event_match = event_match_v[i].reshape(-1,2)

            event_flash_true = event_flash_true_v[i].reshape(-1,n_pmts)
            event_flash_err  = event_flash_err_v[i].reshape(-1,n_pmts)

            flash_err_v  = [None]*len(event_flash)
            flash_true_v = [None]*len(event_flash)
            for j, idx in enumerate(event_ftrue_group[i]):
                flash_true_v[idx] = event_flash_true[j]
            for j, idx in enumerate(event_ferr_group[i]):
                flash_err_v[idx] = event_flash_err[j]
            
            flash_v = []
            for j, flash in enumerate(event_flash):
                flash_v.append(Flash(pe_v=flash,
                    pe_err_v=flash_err_v[j],
                    pe_true_v=flash_true_v[j],
                    idx=event_fidx_v[i][j],
                    time=event_ftime_v[i][j],
                    time_true=event_ftime_true_v[i][j],
                    time_width=event_ftime_width_v[i][j],
                    )
                )

            qcluster_v = []
            start = 0
            for j, ctr in enumerate(event_group):
                qc = QCluster(qpt_v=event_point[start:start+ctr],
                    idx=event_qidx_v[i][j],
                    time=event_qtime_v[i][j],
                    xmin_true=event_qxmin_true_v[i][j])
                qcluster_v.append(qc)
                start = start + ctr
            
            qcluster_vv.append(qcluster_v)
            flash_vv.append(flash_v)
            match_vv.append(event_match)
            
        return (qcluster_vv, flash_vv, match_vv)
        
    
    def write_one(self, qcluster_v: List[QCluster], flash_v: List[Flash], match_v):
        '''
        Write many event to a file with the provided list of QCluster and Flash
        '''
        self.write_many([qcluster_v],[flash_v], [match_v])
        
    def write_many(self,qcluster_vv: List[List[QCluster]], flash_vv: List[List[Flash]], match_vv):
        '''
        Write many event to a file with the provided list of QCluster and Flash
        '''
        if self._mode not in ['w','a']:
            raise ValueError('the dataset is not created in the w (write) nor a (append) mode')
        if len(qcluster_vv) != len(flash_vv):
            raise ValueError(f'len(qcluster_vv) ({len(qcluster_vv)}) != len(flash_vv) ({len(flash_vv)}')
        if len(qcluster_vv) != len(match_vv):
            raise ValueError(f'len(qcluster_vv) ({len(qcluster_vv)}) != len(match_vv) ({len(match_vv)})')
        
        # expand the output count by one for the new entry
        data_index = self._wh_point.shape[0]
        data_count = data_index+len(qcluster_vv)
        for h in [self._wh_point, self._wh_group,
        self._wh_flash, self._wh_flash_err, self._wh_flash_true,
        self._wh_match,
        self._wh_ferr_group, self._wh_ftrue_group,
        self._wh_qidx, self._wh_qtime, self._wh_qxmin_true,
        self._wh_fidx, self._wh_ftime, self._wh_ftime_true, self._wh_ftime_width]:
            h.resize(data_count,axis=0)

        for i in range(len(qcluster_vv)):
        
            qcluster_v = qcluster_vv[i]
            flash_v = flash_vv[i]
            
            ntracks = len(qcluster_v)
            nflash  = len(flash_v)

            # Write points in single flat array, then write grouping separately
            point_v  = []
            for j in range(ntracks):
                point_v.append(qcluster_v[j].qpt_v.cpu().numpy())
            point_group = np.array([pts.shape[0] for pts in point_v])
            point_v = np.concatenate(point_v)
            self._wh_point[data_index] = point_v.flatten()
            self._wh_group[data_index] = point_group
            self._wh_match[data_index] = np.array(match_vv[i]).flatten()
            
            # Write QCluster saclar attributes
            self._wh_qidx [data_index] = np.array([qc.idx  for qc in qcluster_v],dtype=np.int32)
            self._wh_qtime[data_index] = np.array([qc.time for qc in qcluster_v])
            self._wh_qxmin_true[data_index] = np.array([qc.xmin_true for qc in qcluster_v])

            # Write pe-per-pmt in a single flat array. Grouping is fixed so no need to specify.
            photon_v = []
            photon_err_v = []
            photon_true_v = []
            fgroup_err = []
            fgroup_true = []
            for j in range(nflash):

                photon_v.append(flash_v[j].pe_v.cpu().numpy())

                if isinstance(flash_v[j].pe_err_v,torch.Tensor) and len(flash_v[j].pe_err_v):
                    fgroup_err.append(j)
                    photon_err_v.append(flash_v[j].pe_err_v.cpu().numpy())

                if isinstance(flash_v[j].pe_true_v,torch.Tensor) and len(flash_v[j].pe_true_v):
                    fgroup_true.append(j)
                    photon_true_v.append(flash_v[j].pe_true_v.cpu().numpy())

            photon_v = np.concatenate(photon_v)
            if len(photon_err_v)>1:
                photon_err_v = np.concatenate(photon_err_v)
            else:
                photon_err_v = np.array(photon_err_v)
            if len(photon_true_v)>1:
                photon_true_v = np.concatenate(photon_true_v)
            else:
                photon_true_v = np.array(photon_true_v)

            self._wh_flash[data_index] = photon_v.flatten()
            self._wh_flash_true[data_index]  = photon_true_v.flatten()
            self._wh_flash_err[data_index] = photon_err_v.flatten()

            self._wh_ftrue_group[data_index] = np.array(fgroup_true)
            self._wh_ferr_group[data_index]  = np.array(fgroup_err )

            # Write Flash saclar attributes
            self._wh_fidx [data_index] = np.array([f.idx  for f in flash_v],dtype=np.int32)
            self._wh_ftime[data_index] = np.array([f.time for f in flash_v])
            self._wh_ftime_true [data_index] = np.array([f.time_true  for f in flash_v])
            self._wh_ftime_width[data_index] = np.array([f.time_width for f in flash_v])
            
            data_index += 1