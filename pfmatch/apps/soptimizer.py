import os
from time import time

import torch
import yaml
from slar.optimizers import optimizer_factory
from slar.utils import CSVLogger, get_device
from torch.utils.data import DataLoader
from tqdm import tqdm

from pfmatch.algorithms import PoissonMatchLoss, SirenTrack
from pfmatch.apps.toymc import ToyMCDataset


class SOptimizer:
    def __init__(self, cfg:dict):
        """Train the SirenVis model using track data.
        
        This class uses the ToyMCDataset to train the SirenVis model by using
        the model to predict the photo-electron (p.e.) spectrum from a track, and then comparing
        the predicted p.e. spectrum to the true p.e. spectrum using a Poisson loss function.
        
        Several classes are created using the configuration dictionary. Thus, you should include
        the corresponding keys for each class. The following classes are created:
        - `pfamtch.algorithms.SirenTrack`: a wrapper around `SirenVis` (cfg['model'])
        - `pfmatch.apps.toymc.ToyMCDataset`: a dataset for the track data (cfg['data']['dataset'])
            - if generating tracks, `photonlib.PhotonLib` is created (cfg['photonlib'])
            - if generating tracks, `pfmatch.apps.ToyMC` is created (cfg['ToyMC'] and cfg['Detector'])
        - `torch.io.data.DataLoader`: a dataloader for ToyMCDataset (cfg['data']['loader'])
        - `slar.utils.CSVLogger`: a logger for the training process (cfg['logger'])
        - `torch.optim.Optimizer`: an optimizer for the SirenVis model (cfg['train'])
        
        The main method is `train()`, which runs the training loop. See the example notebook
        `Train_SOptimizer.ipynb` for an example configuration and usage.
        """

        if cfg.get('device'):
            self._device = get_device(cfg['device']['type'])
        else:
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # essentials
        self._model = SirenTrack(cfg).to(self._device)
        self._criterion = PoissonMatchLoss().to(self._device)

        # training things
        dataset = ToyMCDataset(cfg)
        self._dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg['data']['loader'])
        self._opt, epoch = optimizer_factory(self._model.parameters(), cfg)
        self._logger = CSVLogger(cfg)
        
        # resume training?
        self.iteration, self.epoch = 0, 0
        ckpt_file = cfg['model'].get('ckpt_file')
        if ckpt_file and cfg['train'].get('resume'):
            import re
            # iteration-{}-epoch-{}.ckpt
            iteration, epoch = re.findall(r'iteration-(\d+)-epoch-(\d+).ckpt',ckpt_file)[0]
            self.iteration, self.epoch = int(iteration), int(epoch)
            print('[SOptimizer] resuming training from iteration',self.iteration,'epoch',self.epoch)

        # training hyperparameters
        train_cfg = cfg.get('train',dict())
        self.epoch_max = train_cfg.get('max_epochs',int(1e20))
        self.iteration_max = train_cfg.get('max_iterations',int(1e20))
        self.save_every_iterations = train_cfg.get('save_every_iterations',-1)
        self.save_every_epochs = train_cfg.get('save_every_epochs',-1)

        # save config
        with open(os.path.join(self._logger.logdir,'train_cfg.yaml'), 'w') as f:
            yaml.safe_dump(cfg, f)


    @property
    def model(self):
        """SirenVis model"""
        return self._model
    
    @property
    def criterion(self):
        """Loss function for the training process (PoissonMatchLoss)"""
        return self._criterion
        
    @property
    def dataloader(self):
        """torch dataloader for the ToyMC dataset"""
        return self._dataloader
    
    @property
    def logger(self):
        """logger for the training process"""
        return self._logger

    @property
    def opt(self):
        """torch optimizer (Adam)"""""
        return self._opt
        
    def step(self, input):
        """Runs a single step of the optimizer.

        Parameters
        ----------
        input : dict
            a dictionary containing a batch of qpt_v, pe_v, and q_sizes, as returned by
            `ToyMCDataset.__getitem__`.
        """

        input['qpt_v'] = input['qpt_v'].to(self._device)
        input['pe_v'] = input['pe_v'].to(self._device)
        input['q_sizes'] = input['q_sizes'].to(self._device)

        # run model
        out = self.model(input)
        target = input['pe_v']
        pred = out['pe_v']

        # compute loss
        loss = self.criterion(pred, target) # todo: weight=1 for now

        # backprop
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        
        return target, pred, loss
    
    def train(self):
        """
        Trains SIREN using the ToyMC dataset.
        """
        twait = time()
        stop_training = False
    
        # epoch loop
        while self.iteration < self.iteration_max and \
              self.epoch < self.epoch_max:  
            # iteration loop (batch loop)
            for batch in tqdm(self.dataloader, desc='Epoch %-3d'%self.epoch, unit='batch'):
                self.iteration += 1
                twait = time() - twait

                # step the model                    
                ttrain = time()
                target, pred, loss = self.step(batch)
                ttrain = time() - ttrain
                
                # log training parameters
                self.logger.record(['iter','epoch','loss','ttrain','twait'],
                                    [self.iteration, self.epoch, loss.item(), ttrain, twait])
                twait = time()

                # step the logger (pe spectrum)
                self.logger.step(self.iteration, target, pred)
                
                # save model params periodically after iterations
                if self.save_every_iterations > 0 and \
                    self.iteration % self.save_every_iterations == 0:
                    self.save()
                    
                if self.iteration_max <= self.iteration:
                    stop_training = True
                    break

            if stop_training:
                break
                        
            self.epoch += 1
            
            # save model params periodically after epochs
            if (self.save_every_epochs*self.epoch) > 0 and self.epoch % self.save_every_epochs == 0:
                self.save(count=self.iteration/len(self.dataloader.dataset))
            
        print('[SOptimizer] Stopped training at iteration',self.iteration,'epochs',self.epoch)
        self.logger.write()
        self.logger.close()


    def save(self, count=None):
        """Saves the model parameters to a checkpoint file."""
        if count is None:
            count = self.epoch

        filename = os.path.join(self.logger.logdir,'iteration-%06d-epoch-%04d.ckpt')
        self.model.save_state(filename % (self.iteration, self.epoch), self.opt, count)
