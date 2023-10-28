import sys, os, importlib, glob, yaml
import torch
import numpy as np
from functools import partial
from pfmatch.datatypes import Flash


def list_available_devices():

    devs=dict(cpu=torch.device('cpu'))

    if torch.cuda.is_available():
        devs['cuda'] = torch.device('cuda:0')
        
    if torch.backends.mps.is_available():
        devs['mps'] = torch.device('mps')

    return devs


def get_device(request):

    devs = list_available_devices()

    if not request in devs:
        print(request,'not supported')
        return None
    else:
        return devs[request]


def import_from(src):
    if src.count('.') == 0:
        module = sys.modules['__main__']
        obj_name = src
    else:
        module_name, obj_name = os.path.splitext(src)
        module = importlib.import_module(module_name)

    return getattr(module, obj_name.lstrip('.'))



def get_config_dir():

    return os.path.join(os.path.dirname(__file__),'../config')


def list_config(full_path=False):

    fs = glob.glob(os.path.join(get_config_dir(), '*.yaml'))

    if full_path:
        return fs

    return [os.path.basename(f)[:-5] for f in fs]


def get_config(name):

    options = list_config()
    results = list_config(True)

    if name in options:
        return results[options.index(name)]

    alt_name = name + '.yaml'
    if alt_name in options:
        return results[options.index(alt_name)]

    print('No data found for config name:',name)
    raise NotImplementedError

def load_config(name:str):

    return yaml.safe_load(open(get_config(name),'r'))

def load_detector_config(name:str):

    if isinstance(name,str):
        return yaml.safe_load(open(get_config(name),'r'))
    elif isinstance(name,dict):
        return yaml.safe_load(open(get_config(name['Detector'])))


def flash_time_integral(flash:Flash, Fractions:list, Taus:list):

    if isinstance(Fractions,float):
        Fractions=[Fractions]
    if isinstance(Taus,float):
        Taus=[Taus]
    assert len(Taus)==len(Fractions), f'Fractions ({Fractions}) and Taus ({Taus}) must have the same lengths'

    assert np.isclose(np.sum(Fractions),1.), f'Fractions must integrate to 1.0: {Fractions}'

    assert np.sum([int(tau>0.) for tau in Taus]) == len(Taus), f'Taus must be all positive value: {Taus}'

    if flash.time_width is None or np.isnan(flash.time_width) or np.isinf(flash.time_width):
        raise RunTimeError(f'flash.time_width is invalid {flash.time_width}')

    integral_factor = 0.
    for i in range(len(Fractions)):
        integral_factor += Fractions[i] * (1. - np.exp(-1. * flash.time_width / Taus[i]))

    return integral_factor

def partial_flash_time_integral(cfg:dict):

    return partial(flash_time_integral, **cfg['FlashTimeIntegral'])
