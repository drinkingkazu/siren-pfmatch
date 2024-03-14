from __future__ import annotations

import itertools
import time
import torch
import numpy as np
from photonlib import PhotonLib
from slar.nets import SirenVis
from pfmatch.datatypes import FlashMatch
from pfmatch.apps import XOptimizer


class MOptimizer:
    
    def __init__(self, cfg:dict, plib: PhotonLib | MultiVis | SirenVis | MultiVis):
        
        self._plib = plib
        this_cfg = cfg.get('MOptimizer',dict())
        self.verbose = this_cfg.get('Verbose',False)
        self.prefilter = this_cfg.get('Prefilter',False)
        self.prefilter_topk = this_cfg.get('PrefilterTopK',2)
        self.prefilter_loss = this_cfg.get('PrefilterLoss',200)
        self._model = XOptimizer(cfg,plib)
        
    @property
    def model(self):
        return self._model
        
    @property
    def plib(self):
        return self._plib
    
    def select_topk_2d(self, mat : torch.Tensor, 
                       topk : int, 
                       loss_threshold : float, 
                       mode='both'):

        result=[]

        if mode in ['row','both']:
            for row,col_v in enumerate(torch.argsort(mat,dim=1)):
                pairs=[]
                for d in col_v:
                    col=d.item()
                    if mat[row,col] > loss_threshold:
                        break
                    pairs.append([row,col])
                    if len(pairs) >= topk:
                        break
                result += pairs
        if mode in ['col','both']:
            for col,row_v in enumerate(torch.argsort(mat,dim=0).T):
                pairs=[]
                for d in row_v:
                    row=d.item()
                    if mat[row,col] > loss_threshold:
                        break
                    pairs.append([row,col])
                    if len(pairs) >= topk:
                        break
                result += pairs

        return result
        
    def scan_and_prefilter(self, input):
        match = FlashMatch(len(input.qcluster_v),
                           len(input.flash_v),
                           input.flash_v[0].pe_v.shape[0])
        pairs = list(itertools.product(input.qcluster_v, input.flash_v))
        
        pairs_ids = []
        track_id, flash_id = 0, 0
        for qcluster, flash in pairs:
            prefit   = self._model.scan_loss(qcluster.qpt_v,flash.pe_v)
            loss     = prefit['loss_init']
            reco_x   = prefit['dx_init'  ]
            reco_pe  = prefit['pe_init'  ]
            duration = prefit['tspent'   ]
            match.loss_matrix[track_id, flash_id] = loss
            match.reco_x_matrix[track_id, flash_id] = reco_x
            match.reco_pe_matrix[track_id, flash_id] = reco_pe
            match.duration[track_id, flash_id] = duration
            pairs_ids.append((track_id,flash_id))
            if flash_id < len(input.flash_v) - 1:
                flash_id += 1
            else:
                track_id += 1
                flash_id = 0
                
        # Now scan the results and keep the top-N
        topk_pairs = self.select_topk_2d(match.loss_matrix, 
                                         topk=self.prefilter_topk,
                                         loss_threshold=self.prefilter_loss,
                                         mode='both')

        flags = [([row,col] in topk_pairs) for row,col in pairs_ids]

        if self.verbose:
            print(f'[MOptimizer:prefilter] selected {np.sum(flags)} out of {len(pairs)}')
            print(f'[MOptimizer:prefilter] time taken for pre-filtering {match.duration.sum().item()} [s]')
            
        return match, pairs, flags
        
    
    def fit(self, input):
        """
        Run flash matching on flashmatch input
        --------
        Arguments
          flashmatch_input: FlashMatchInput object
        --------
        Returns
          FlashMatch object storing the result of the match
        """        
        

        if self.prefilter:
            match, pairs, flags = self.scan_and_prefilter(input)
        else:
            match = FlashMatch(len(input.qcluster_v),
                               len(input.flash_v),
                               input.flash_v[0].pe_v.shape[0])
            pairs = list(itertools.product(input.qcluster_v, input.flash_v))
            flags = [True]*len(pairs)


        track_id, flash_id = 0, 0           
        for idx, (qcluster, flash) in enumerate(pairs):
            if flags[idx]:
                dx_init=None
                if self.prefilter:
                    dx_init = match.reco_x_matrix[track_id, flash_id].item()
                fit_res  = self._model.fit(qcluster,flash,dx_init)['fit']
                loss     = fit_res['loss_best']
                reco_x   = fit_res['dx_best'  ]
                reco_pe  = fit_res['pe_best'  ]
                duration = fit_res['tspent'] 
                match.loss_matrix[track_id, flash_id] = loss
                match.reco_x_matrix[track_id, flash_id] = reco_x
                match.reco_pe_matrix[track_id, flash_id] = reco_pe
                match.duration[track_id, flash_id] += duration
            if flash_id < len(input.flash_v) - 1:
                flash_id += 1
            else:
                track_id += 1
                flash_id = 0

        #match.bipartite_match()
        return match

