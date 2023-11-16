import os
from time import time

import torch
import yaml
from slar.nets import SirenVis, WeightedL2Loss
from slar.optimizers import optimizer_factory
from slar.utils import CSVLogger, get_device
from torch.utils.data import DataLoader, Dataset

from pfmatch.apps import ToyMC
from pfmatch.utils import load_detector_config
from photonlib import PhotonLib
from tqdm import tqdm


class ToyMCDataset(Dataset):
    """
    ToyMC track generation in the form of torch Dataset for training Siren
    """
    
    def __init__(self, cfg):
        """Constructor

        Parameters
        ----------
        cfg : dict
            model configuration.
        """

        self._plib = PhotonLib.load(cfg)
        self._toymc = ToyMC(cfg, load_detector_config(cfg), self._plib)
        
        this_cfg = cfg.get('data')
        self._num_tracks = this_cfg['dataset']['size']
        
        # todo: vis weights like in PhotonLibDataset?
        
    def __len__(self):
        return self._num_tracks

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int
            index of the track to generate

        Returns
        -------
        tuple
            (qcluster, flash) pair
        """
        track = self._toymc.gen_trajectories(1)
        qpt_v = self._toymc.make_qpoints([track])
        pe_v, *__ = self._toymc.make_photons(qpt_v)   
        
        # normalize qpt coords to [-1,1]     
        qpt_v[:,:3] = self._plib.meta.norm_coord(qpt_v[:,:3])
        
        output = dict(
            qpt_v=qpt_v,
            pe_v=pe_v,
            # todo: weight?
        )
        
        return output
    
class SirenTrack(SirenVis):
    def __init__(self, cfg : dict, ckpt_file : str = None):
        # this should work with siren's already trained with voxel data!
        super().__init__(cfg, ckpt_file)

    def forward(self, x):
        coord, pe = x[:,:3], x[:,3]
        
        out = super().forward(coord)       # vis in log scale
        out = super()._inv_xform_vis(out)  # vis in linear scale

        # todo: apply poisson like in ToyMC.make_photons?
        pe_pred = torch.sum(out*pe.unsqueeze(-1), axis=0)

        return pe_pred


class SOptimizer:
    def __init__(self, cfg:dict, plib: PhotonLib | SirenVis):
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if cfg.get('device'):
            self._device = get_device(cfg['device']['type'])
        
        # essentials
        self._plib = plib
        self._model = SirenTrack(cfg).to(self._device)
        self._criterion = WeightedL2Loss()
        
        # training things
        dataset = ToyMCDataset(cfg)
        self._dataloader = DataLoader(dataset, **cfg['data']['loader'])
        self._opt, self.epoch = optimizer_factory(self._model.parameters(), cfg)
        self._logger = CSVLogger(cfg)
        self.iteration, self.epoch = 0, 0

        # training hyperparameters
        train_cfg = cfg.get('train',dict())
        self.epoch_max = train_cfg.get('max_epochs',int(1e20)),
        self.iteration_max = train_cfg.get('max_iterations',int(1e20)),
        self.save_every_iterations = train_cfg.get('save_every_iterations',-1),
        self.save_every_epochs = train_cfg.get('save_every_epochs',-1)  

        # save config
        with open(os.path.join(self._logger.logdir,'train_cfg.yaml'), 'w') as f:
            yaml.safe_dump(cfg, f)


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
    def dataloader(self):
        return self._dataloader
    
    @property
    def logger(self):
        return self._logger

    @property
    def opt(self):
        return self._opt
        
    def step(self, input):
        """fit the model to the input data

        Parameters
        ----------
        input : torch.Tensor
            modified qpt_v tensor with normalized coordinates and linear pe counts.
            it is assumed that the inputs are already shuffled and batched, and on
            the correct device.
        """

        input = input['qpt_v'].to(self._device)
        target = input['pe_v'].to(self._device)

        # run model
        pred = self.model(input)
        # compute loss
        loss = self.criterion(pred, target) # todo: weight=1 for now

        # backprop
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        
        return target, pred, loss
    
    def train(self):
        twait = time()
        stop_training = False
    
        while self.iteration < self.iteration_max and \
              self.epoch < self.epoch_max:  

            for data in tqdm(self.dataloader, desc='Epoch %-3d'%self.epochs):
                self.iteration += 1
                twait = time() - twait

                # step the model                    
                ttrain = time()
                target, pred, loss = self.step(data)
                ttrain = time() - ttrain
                
                # log training parameters
                self.logger.record(['iter','epoch','loss','ttrain','twait'],
                                    [self.iteration, self.epoch, loss.item(), loss, ttrain, twait])
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
            
        print('[train] Stopped training at iteration',self.iteration,'epochs',self.epoch)
        self.logger.write()
        self.logger.close()


    def save(self, count=None):
        if count is None:
            count = self.iteration

        filename = os.path.join(self.logger.logdir,'iteration-%06d-epoch-%04d.ckpt')
        self.model.save_state(filename % (self.iteration, self.epoch), self.opt, count)
