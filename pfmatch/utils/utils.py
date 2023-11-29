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
    
def load_toymc_config(name:str):

    if isinstance(name,str):
        return yaml.safe_load(open(get_config(name),'r'))
    elif isinstance(name,dict):
        return yaml.safe_load(open(get_config(name['ToyMC'])))

def flash_time_integral(flash:Flash, Fractions:list, Taus:list):

    if isinstance(Fractions,float):
        Fractions=[Fractions]
    if isinstance(Taus,float):
        Taus=[Taus]
    assert len(Taus)==len(Fractions), f'Fractions ({Fractions}) and Taus ({Taus}) must have the same lengths'

    assert np.isclose(np.sum(Fractions),1.), f'Fractions must integrate to 1.0: {Fractions}'

    assert np.sum([int(tau>0.) for tau in Taus]) == len(Taus), f'Taus must be all positive value: {Taus}'

    if flash.time_width is None or np.isnan(flash.time_width) or np.isinf(flash.time_width):
        raise RuntimeError(f'flash.time_width is invalid {flash.time_width}')

    integral_factor = 0.
    for i in range(len(Fractions)):
        integral_factor += Fractions[i] * (1. - np.exp(-1. * flash.time_width / Taus[i]))

    return integral_factor

def partial_flash_time_integral(cfg:dict):

    return partial(flash_time_integral, **cfg['FlashTimeIntegral'])

def generate_unbounded_tracks(N, boundary, extension_factor=0.2, rng=np.random):
    """
    Generates N tracks within a given 3D boundary with an extension factor to improve edge coverage.
    
    Parameters:
    - N (int): Number of tracks to generate.
    - boundary (tuple): Tuple of tuples defining the min and max of the range in 3D ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
    - extension_factor (float): Percentage to extend the boundary by for initial point generation.
    
    Returns:
    - np.array: An array of shape (N, 2, 3) containing the start and end points of each track.
    """
    
    # Calculate extended boundary
    extended_boundary = [(b[0] - (b[1] - b[0]) * extension_factor, 
                          b[1] + (b[1] - b[0]) * extension_factor) for b in boundary]

    # Generate random tracks within the extended boundary
    tracks = np.asarray([rng.uniform(*extended_boundary[i], size=(2, N)) for i in range(3)])
    
    return tracks.T

def find_intersection_with_boundary(boundary, point_inside, point_outside):
    """
    Given a boundary and two points of a line segment that crosses the boundary of a cube,
    this function finds the intersection point of the line segment with the boundary.

    Parameters:
    - boundary (tuple): ((xmin, xmax), (ymin, ymax), (zmin, zmax)) defining the cube boundary.
    - point_inside (np.array): The point inside the boundary (good point).
    - point_outside (np.array): The point outside the boundary (bad point).

    Returns:
    - np.array: The intersection point with the boundary.
    """

    # Calculate the direction vector from the outside point to the inside point
    direction = point_inside - point_outside

    # Initialize intersection point
    intersection_point = point_outside

    # Check for intersection with each face of the boundary box
    for i, (min_val, max_val) in enumerate(boundary):
        # Skip if there's no movement in this direction
        if direction[i] == 0:
            continue
        
        # Calculate parameter t for intersection with the near and far planes in each axis
        for bound in [min_val, max_val]:
            t = (bound - point_outside[i]) / direction[i]
            if t < 0 or t > 1:
                continue  # The intersection is not between the points
            
            # Find the intersection point
            intersection_test = point_outside + t * direction
            
            # Check if the intersection point is within the boundary on the other two axes
            if all(min_val <= intersection_test[j] <= max_val for j, (min_val, max_val) in enumerate(boundary) if j != i):
                intersection_point = intersection_test
                return intersection_point

    return intersection_point

def is_outside(point, boundary):
    """
    Checks if a point is outside the boundary.
    
    Parameters:
    - point (np.array): The point to check.
    - boundary (tuple): ((xmin, xmax), (ymin, ymax), (zmin, zmax)) defining the cube boundary.
    
    Returns:
    - bool: True if the point is outside the boundary, False otherwise.
    """
    return any([point[i] < boundary[i][0] or point[i] > boundary[i][1] for i in range(3)])

class CSVLogger:
    '''
    Logger class to store training progress in a CSV file.
    '''

    def __init__(self,cfg):
        '''
        Constructor

        Parameters
        ----------
        cfg : dict
            A collection of configuration parameters. `dir_name` and `file_name` specify
            the output log file location. `analysis` specifies analysis function(s) to be
            created from the analysis module and run during the training.
        '''
        
        log_cfg = cfg.get('logger',dict())
        self._logdir  = self.make_logdir(log_cfg.get('dir_name','logs'))
        self._logfile = os.path.join(self._logdir, cfg.get('file_name','log.csv'))
        self._log_every_nsteps = log_cfg.get('log_every_nsteps',1)

        print('[CSVLogger] output log file:',self._logfile)
        print(f'[CSVLogger] recording a log every {self._log_every_nsteps} steps')
        self._fout = None
        self._str  = None
        self._dict = {}
        
        self._analysis_dict={}
        
        for key, kwargs in log_cfg.get('analysis',dict()).items():
            print('[CSVLogger] adding analysis function:',key)
            kwargs = kwargs if kwargs else dict()
            self._analysis_dict[key] = partial(getattr(importlib.import_module('pfmatch.analysis'),key), **kwargs)
        
    @property
    def logfile(self):
        return self._logfile
    
    @property
    def logdir(self):
        return self._logdir
        
    def make_logdir(self, dir_name):
        '''
        Create a log directory

        Parameters
        ----------
        dir_name : str
            The directory name for a log file. There will be a sub-directory named version-XX where XX is
            the lowest integer such that a subdirectory does not yet exist.

        Returns
        -------
        str
            The created log directory path.
        '''
        versions = [int(d.split('-')[-1]) for d in glob.glob(os.path.join(dir_name,'version-[0-9][0-9]'))]
        ver = 0
        if len(versions):
            ver = max(versions)+1
        logdir = os.path.join(dir_name,'version-%02d' % ver)
        os.makedirs(logdir)

        return logdir

    def record(self, keys : list, vals : list):
        '''
        Function to register key-value pair to be stored

        Parameters
        ----------
        keys : list
            A list of parameter names to be stored in a log file.

        vals : list
            A list of parameter values to be stored in a log file.
        '''
        for i, key in enumerate(keys):
            self._dict[key] = vals[i]
            
    def step(self, iteration, label=None, pred=None):
        '''
        Function to take a iteration step during training/inference. If this step is
        subject for logging, this function 1) runs analysis methods and 2) write the
        parameters registered through the record function to an output log file.

        Parameters
        ----------
        iteration : int
            The current iteration for the step. If it's not modulo the specified steps to
            record a log, the function does nothing.

        label : torch.Tensor
            The target values (labels) for the model run for training/inference.

        pred : torch.Tensor
            The predicted values from the model run for training/inference.


        '''
        if not iteration % self._log_every_nsteps == 0:
            return
        
        if not None in (label,pred):
            for key, f in self._analysis_dict.items():
                self.record([key],[f(label,pred)])
        self.write()

    def write(self):
        '''
        Function to write the key-value pairs provided through the record function
        to an output log file.
        '''
        if self._str is None:
            self._fout=open(self._logfile,'w')
            self._str=''
            for i,key in enumerate(self._dict.keys()):
                if i:
                    self._fout.write(',')
                    self._str += ','
                self._fout.write(key)
                self._str+='{:f}'
            self._fout.write('\n')
            self._str+='\n'
        self._fout.write(self._str.format(*(self._dict.values())))
        self.flush()
        
    def flush(self):
        '''
        Flush the output file stream.
        '''
        if self._fout: self._fout.flush()

    def close(self):
        '''
        Close the output file.
        '''
        if self._str is not None:
            self._fout.close()
