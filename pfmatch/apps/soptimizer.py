import time
import torch
import numpy as np
from photonlib import PhotonLib
from slar.nets import SirenVis
from pfmatch.datatypes import QCluster
from pfmatch.algorithms import PoissonMatchLoss, FlashHypothesis


class SOptimizer:
    
    def __init__(self, cfg:dict, plib: PhotonLib | SirenVis):
        
        self._plib = plib
        this_cfg = cfg['SOptimizer']


    @property
    def model(self):
        return self._model
    
    @property
    def criterion(self):
        return self._criterion 
        
    @property
    def plib(self):
        return self._plib
        
    def fit(self, input):
        pass