from __future__ import annotations

import copy
import time
import torch
import numpy as np
from photonlib import PhotonLib
from slar.nets import SirenVis
from pfmatch.datatypes import QCluster
from pfmatch.algorithms import PoissonMatchLoss, FlashHypothesis
from pfmatch.utils import partial_flash_time_integral

class XOptimizer:
    
    def __init__(self, cfg:dict, plib: PhotonLib | SirenVis):
        
        self._plib = plib
        this_cfg = cfg['XOptimizer']
        self.max_iterations  = this_cfg.get('MaxIteration',        500  )
        self.initial_lr      = this_cfg.get('InitLearningRate',    1.0  )
        self.stop_patience   = this_cfg.get('StopPatience',        20   )
        self.stop_xmin_delta = this_cfg.get('StopDeltaXMin',       0.1  )
        self.stop_loss_delta = this_cfg.get('StopDeltaLoss',       0.001)
        self.loss_scan_step  = this_cfg.get('LossScanStepSize',    -1.  )
        self.correct_pe_time = this_cfg.get('CorrectTimeWidth',    False)
        self.verbose         = this_cfg.get('Verbose',             False)

        self.pe_correction_fn = lambda v : 1
        if self.correct_pe_time:
            self.pe_correction_fn = partial_flash_time_integral(cfg)

        
        self.lrs_type = None
        self.lrs_args = None
        lrs = this_cfg.get('LearningRateScheduler',dict())
        if lrs.get('Name'):
            if not hasattr(torch.optim.lr_scheduler,lrs['Name']):
                raise RunTimeError(f'Learning rate scheduler not available: {lrs["Name"]}')
                
            self.lrs_type = getattr(torch.optim.lr_scheduler,lrs['Name'])
            self.lrs_args = lrs.get('Parameters')        
        
        self._model = FlashHypothesis(cfg,self.plib)
        self._criterion = PoissonMatchLoss()
        
        # Containers for the optimization output
        self._loss_history = []
        self._xmin_history = []
        self._time_spent   = 0.
        self._hypothesis_history = None

    @property
    def model(self):
        return self._model
    
    @property
    def criterion(self):
        return self._criterion 
        
    @property
    def plib(self):
        return self._plib
    
    @property
    def loss_history(self):
        return self._loss_history
    
    @property
    def xmin_history(self):
        return self._xmin_history
    
    @property
    def time_spent(self):
        return self._time_spent
    
    @property
    def hypothesis_history(self):
        return self._hypothesis_history[:len(self.loss_history)]
    
    @property
    def loss(self):
        best_idx = np.argmin(self._loss_history)
        return self._loss_history[best_idx]
    
    @property
    def xmin(self):
        best_idx = np.argmin(self._loss_history)
        return self._xmin_history[best_idx]

    @property
    def hypothesis(self):
        best_idx = np.argmin(self._loss_history)
        return self._hypothesis_history[best_idx]
    

    def scan_loss(self, input:torch.Tensor, target:torch.Tensor):

        # Calculate the range of dx value to scan
        dx_min, dx_max = self.model.dx_range(input)
        dx_range = dx_max - dx_min
        if self.verbose:
            print('[XOptimizer:scan_loss] dx range %.2f => %.2f ... span %.2f' % (dx_min, dx_max, dx_range))
        if dx_range < (self.loss_scan_step*2):
            if self.verbose:
                print('[XOptimizer:scan_loss] dx_range too short! use initial dx=0')
            return 0.

        # Scan the loss over num_steps, starting from margin in the step of loss_scan_step size
        num_steps = int(dx_range/self.loss_scan_step)
        margin = (dx_range - self.loss_scan_step * num_steps)/2.
        tstart = time.time()
        loss_v = []
        dx_v   = []
        hypothesis_v = torch.empty(size=(num_steps+1,len(target)),
            dtype=target.dtype,
            device=target.device)
        with torch.no_grad():

            for i in range(num_steps+1):
                dx = margin + i * self.loss_scan_step + dx_min
                self.model.dx = dx
                pred = self.model(input)
                loss = self.criterion(pred,target).item()
                loss_v.append(loss)
                dx_v.append(dx)
                hypothesis_v[i][:] = pred[:]

        if self.verbose:
            print('Scanned dx v.s. loss')
            for i in range(len(loss_v)):
                print('[XOptimizer:scan_loss] dx %.2f xmin %.2f loss %.2f' % (dx_v[i],dx_v[i]+input[:,0].min().item(),loss_v[i]))

        idx = np.argmin(loss_v)
        if self.verbose:
            print('[XOptimizer:scan_loss] choosing the initial dx %.2f xmin %.2f' % (dx_v[idx],dx_v[i]+input[:,0].min().item()))

        tspent = time.time()-tstart
        return loss_v[idx], dx_v[idx], hypothesis_v[idx], tspent
        
    def fit(self, qcluster:QCluster, flash:Flash, dx:float = None):
        """
        Run gradient descent model on input
        --------
        Arguments
          input: qcluster input as tensor
          target: flash target as tensor
          dx0: initial xshift in cm
          dx_min: miminal allowed value of dx in cm
          dx_max: maximum allowed value of dx in cm
        --------
        Returns
          loss, reco_x, reco_pe
        """

        input  = qcluster.qpt_v
        target = flash.pe_v

        # Initialize the result containers
        self._loss_history = []
        self._xmin_history = []
        self._time_spent   = 0.

        # Initialize pe_history, a bit more treatment
        if self._hypothesis_history is None:
            self._hypothesis_history = torch.zeros(size=(self.max_iterations,len(target)),
                dtype=target.dtype, device=target.device)
        self._hypothesis_history[:,:]=0.

        # if requested to correct for the time width, perform here
        if self.correct_pe_time:
            target = copy.deepcopy(flash.pe_v)
            fraction = self.pe_correction_fn(flash)
            if self.verbose:
                print('[XOptimizer:fit] PE time integral',fraction,'correction factor',1./fraction)
            target[:] /= fraction

        #model.to(device)

        # Perform the initial loss scan if configured (loss_scan_step>0)
        if dx is not None:
            self.model.dx = dx
        elif self.loss_scan_step > 0:
            self.model.dx = self.scan_loss(input,target)[1]
        else:
            self.model.dx = 0.

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr)
        
        scheduler = None
        if self.lrs_type:
            scheduler = self.lrs_type(optimizer, **self.lrs_args)

        tstart = time.time()
        best_loss = 1.e+20
        best_xmin = 1.e+20
        best_loss_found = False
        xmin_delta_large = False
        stuck_count = 0
        msg_fmt = '[XOptimizer:fit] iter %03d lr %.4f xmin %.3f loss %.6f stuck ctr %03d'
        
        for i in range(self.max_iterations):
            
            pred = self.model(input)
            loss = self.criterion(pred, target)
            self._loss_history.append(loss.item())
            self._xmin_history.append(self.model.dx + input[:,0].min())
            self._hypothesis_history[i][:] = pred[:]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step(loss)

            if (best_loss - self._loss_history[-1]) > self.stop_loss_delta:
                best_loss = self._loss_history[-1]
                best_xmin = self._xmin_history[-1]
                stuck_count = 0
                best_loss_found = True
            else:
                best_loss_found = False
                
            if abs(best_xmin - self._xmin_history[-1]) > self.stop_xmin_delta:
                xmin_delta_large = True
                stuck_count = 0
            else:
                xmin_delta_large = False
                stuck_count += 1
                
            #if i>1 and abs(self._xmin_history[-1] - self._xmin_history[-2]) < self.stop_xmin_delta:
            #    stuck_count += 1
            #else:
            #    stuck_count = 0
                
            if self.verbose:
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                msg = msg_fmt % (i,
                                 lr,
                                 self._xmin_history[-1],
                                 self._loss_history[-1],
                                 stuck_count)
                if best_loss_found:
                    msg += '... best loss found'
                if xmin_delta_large:
                    msg += f'... xmin delta large {abs(best_xmin - self._xmin_history[-1])}'
                print(msg)
                
            if stuck_count >= self.stop_patience:
                break

        self._time_spent = time.time() - tstart

        #return the best loss
        best_idx = np.argmin(self._loss_history)
        return (self._loss_history[best_idx],
            self._xmin_history[best_idx],
            self._hypothesis_history[best_idx],
            self._time_spent)
