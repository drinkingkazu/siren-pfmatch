import torch

from tqdm.auto import tqdm
from slar.nets import SirenVis, MultiVis
from pfmatch.utils import scheduler_factory, CSVLogger
from pfmatch.algorithms import PoissonMatchLoss
from pfmatch.__future__.io import loader_factory

class SOptimizer:
    def __init__(self, cfg, device=None, resume=True):
        if 'multivis' in cfg:
            print('[SOptimizer] Using MultiVis')
            model = MultiVis(cfg)
        elif 'model' in cfg:
            print('[SOptimizer] Using SirenVis')
            model = SirenVis(cfg)
        else:
            raise NotImplementedError('No "multivis" or "model" in cfg')
        
        self._model = model.to(device)
        
        train_cfg = cfg.get('train')
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), **cfg.get('optimizer', {})
        )
        
        self._scheduler = scheduler_factory(
            self._optimizer, cfg.get('scheduler', {})
        )
        
        self._dataloader = loader_factory(cfg)
        self._logger = CSVLogger(cfg)
        
        self.criterion = PoissonMatchLoss()

        self.epoch = 0
        self.n_iter = 0
        self.epoch_max = train_cfg.get('max_epochs', int(1e20))
        self.iter_max = train_cfg.get('max_iterations', int(1e20))
        self.save_every_iters = train_cfg.get('save_every_iterations', -1)
        self.save_every_epochs = train_cfg.get('save_every_epochs', -1)
    
    
    @property
    def device(self):
        return self._model.device
    
    def step(self, batch):
        device = self.device
        
        # prepare input to device
        qpts = torch.cat(batch['qclusters']).to(device)
        target = torch.stack(batch['flashes']).to(device)
        sizes = list(map(len, batch['qclusters']))
        
        # FIXME
        q = qpts[:,3]
        vis_q = self._model.visibility(qpts[:,:3]) * q.unsqueeze(-1)    
    
        pred = torch.stack([
            arr.sum(axis=0) for arr in torch.split(vis_q, sizes)
        ])
        del vis_q
        
        # target pe from flashes
        loss = self.criterion(pred, target)
        
        return pred, loss
    
    def train(self):
        while self.epoch < self.epoch_max and self.n_iter < self.iter_max:
            self.train_one_epoch()
        self._logger.write()
        self._logger.close()
    
    def train_one_epoch(self):
        from time import time
        
        t_start = time()
        for batch in tqdm(self._dataloader, desc=f'epoch {self.epoch}'):
            t_wait = time() - t_start
            
            t_start = time()
            pred, loss_pred = self.step(batch)
            
            loss = loss_pred.mean()
            loss.backward()
            self._optimizer.step()
            
            #TODO(2024-03-22 kvt) update scheduler
            
            t_train = time() - t_start
            
            # Logger
            log_dict = {
                'iter': self.n_iter,
                'epoch': self.epoch,
                'loss': loss.item(),
                'twait': t_wait,
                'ttrain' : t_train,
            }
            self.log(log_dict)
            self._logger.step(self.n_iter)
            
            if (self.save_every_iters > 0
                and self.n_iter % self.save_every_iters == 0):
                self.save()
            
            self.n_iter += 1
            if self.n_iter >= self.iter_max:
                break
            t_start = time()
            
        if (self.save_every_epochs > 0 
                and self.epochs % self.save_every_epochs == 0):
                self.save()
                

                
        self.epoch += 1
        
    def log(self, log_dict):
        cols = list(log_dict.keys())
        values = list(log_dict.values())
        self._logger.record(cols, values)
        
    def save(self):
        import os
        fpath = os.path.join(
            self._logger.logdir,
            f'iteration-{self.n_iter:06}-epoch-{self.epoch:03}.ckpt'
        )
        self._model.save_state(fpath, self._optimizer)
        
    def load(self):
        pass
