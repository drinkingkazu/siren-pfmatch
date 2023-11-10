from pfmatch.datatypes.qcluster import QCluster
from tests.fixtures import rng

import numpy as np
import torch
import pytest

from pfmatch.algorithms import LightPath
from tests.fixtures import rng

@pytest.fixture
def configs(rng):
    cfg = {'LightPath': {'SegmentSize': rng.random()}}
    detector_specs = {'MIPdEdx': rng.random()+1, 'LightYield': 2e4*rng.random()+5e3} 
    return cfg, detector_specs

@pytest.fixture
def lightpath(configs):
    return LightPath(*configs)

def test_lightpath_configure(rng, configs):
    cfg, detector_specs = configs
    lightpath = LightPath(cfg, detector_specs)
    assert lightpath.dEdxMIP == detector_specs['MIPdEdx']
    assert lightpath.light_yield == detector_specs['LightYield']
    assert lightpath.gap == cfg['LightPath']['SegmentSize'] 

    # without any config
    LightPath()
    

def test_lightpath_fill_qcluster(rng, lightpath):
    # test for list input
    random = rng.random()*99 + 1
    
    pt1 = [0,0,0]
    pt2 = [0,random,0]
    lightpath.segment_to_qpoints(pt1, pt2)
    
    # test for numpy array input
    pt1 = np.array([0,0,0])
    pt2 = np.array([0,random,0])
    lightpath.segment_to_qpoints(pt1, pt2)
    
    # test for torch tensor input
    pt1 = torch.tensor([random,0,random], dtype=torch.float32)
    pt2 = torch.tensor([0,random,0], dtype=torch.float32)
    lightpath.segment_to_qpoints(pt1, pt2)

    # test output:
    #   - charge must be positive
    #   - inputs that are all in the same direction should result in outputs
    #     in the same direction
    randx = rng.random()*99 + 1
    randy = rng.random()*99 + 1
    randz = rng.random()*99 + 1
    track_length1 = np.array([[0, 0, 0], [randx, randy, randz]])
    
    # create a track with more than two points
    length = int(rng.random()*98)+2
    track_lengthN = np.array([[0, 0, 0]] + [[i*randx, i*randy, i*randz] for i in range(1, length)])

    for track in [track_length1, track_lengthN]:
        # create a qcluster from the track
        qcluster = None
        for i in range(len(track)-1):
            if qcluster is None:
                qcluster = QCluster(lightpath.segment_to_qpoints(track[i], track[i+1]))
                continue
            
            qcluster += QCluster(lightpath.segment_to_qpoints(track[i], track[i+1]))

        assert all(qcluster.qpt_v[:, 3] >= 0), "Charge must be positive"

        directions = [arr/np.linalg.norm(arr) for arr in qcluster.qpt_v[:, :3]]
        cumdot = [torch.dot(directions[i], directions[i-1]).item() for i in range(len(directions)-1)]
        init_direction = [(track[1]-track[0])/np.linalg.norm(track[1]-track[0])]
        cumdot = [np.dot(init_direction, directions[0].cpu().numpy()).item()] + cumdot
        assert np.allclose(cumdot, np.ones_like(cumdot)), "Segments not in uniform direction"
        
    # test for zero length track
    qcluster = lightpath.segment_to_qpoints(np.array([0,0,0]), np.array([0,0,0]))
    assert qcluster.shape == (1,4)

def test_lightpath_fill_qcluster_from_track(rng, lightpath):
    randx = rng.random()*99 + 1
    randy = rng.random()*99 + 1
    randz = rng.random()*99 + 1
    track_length1 = np.array([[0, 0, 0], [randx, randy, randz]])
    
    # create a track with more than two points
    length = int(rng.random()*98)+2
    track_lengthN = np.array([[0, 0, 0]] + [[i*randx, i*randy, i*randz] for i in range(1, length)])
    for i, track in enumerate([track_length1, track_lengthN]):
        qcluster = lightpath.track_to_qpoints(track)
        
        assert all(qcluster[:, 3] >= 0), "charge must be positive"
        assert np.allclose(qcluster[0, :3], track[0], atol=lightpath.gap), \
            "expected first point to be the first point of the track, within the segment size"
        assert np.allclose(qcluster[-1, :3], track[-1], atol=lightpath.gap), \
            "expected last point to be the last point of the track, within the segment size"
        
    # test track of 0 length
    with pytest.raises(AssertionError):
        lightpath.track_to_qpoints([[0,0,0]])