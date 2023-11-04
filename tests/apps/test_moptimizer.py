from pfmatch.apps.toymc import ToyMC
import pytest
import torch
import os
import yaml
from copy import deepcopy

from pfmatch.apps import MOptimizer
from pfmatch.datatypes import FlashMatchInput

# not necessary but helpful for IDEs
from tests.fixtures import torch_rng, plib, siren, fake_flashmatch_data

""" ------------------------------- fixtures ------------------------------- """
@pytest.fixture
def rand(torch_rng):
    return lambda *args, **kwargs: torch.rand(*args, generator=torch_rng, **kwargs)

@pytest.fixture
def mock_input(fake_flashmatch_data, plib, siren):
    out = FlashMatchInput()
    
    xmins, xmaxs = [], []
    for lib in [plib, siren]:
        xmins.append(lib.meta.ranges[0,0].item())
        xmaxs.append(lib.meta.ranges[0,1].item())
    plib_xmin = max(xmins)
    plib_xmax = min(xmaxs)
    qclusters, flashes = fake_flashmatch_data[0]
    for qcluster in qclusters:
        qcluster.qpt_v[:,0] = plib_xmin + 0.5 * (plib_xmax - plib_xmin) * qcluster.qpt_v[:,0]
        qcluster.qpt_v = qcluster.qpt_v
    out.qcluster_v = qclusters
    out.flash_v = flashes
    return out

@pytest.fixture
def cfg():    
    # create a mock config for testing
    # - note: no config options for XOptimizer in here
    cfg = """
    MOptimizer:
        Verbose:       True
        Prefilter:     True
        PrefilterTopK: 2
        PrefilterLoss: 200

    ToyMC:
        TimeAlgo: "random" # random, periodic, same
        TrackAlgo: "top-bottom" # random, top-bottom
        PeriodTPC: [-1300.,1300] # in micro-second, [-340-965(drift time),1698.4-340]
        PeriodPMT: [-1000., 1000.] # in micro-second, [-1150, 3450-1150]
        PEVariation: 0.00 # channel-to-channel (PMT-wise) variation in (fake) reco PE amount
        LightYieldVariation: 0.00  # light yield variation per point in QCluster_t
        PosXVariation: 0.0 # total x position variation in cm
        TruncateTPC: 0 # 1=truncate TPC trajectory mimicing readout effect
        NumTracks: 5
        NumpySeed: 1
        NeutrinoTime: 0.
        
    XOptimizer:
        Verbose: False
    """
    
    return dict(yaml.safe_load(cfg))
    
@pytest.fixture
def MOptimizers(cfg, plib, siren):
    cfg_no_prefilter = deepcopy(cfg)
    cfg_no_prefilter['MOptimizer']['Prefilter'] = False
    dicts = [cfg, cfg_no_prefilter]
    plibs = [plib, siren]
    return [MOptimizer(d, p) for d in dicts for p in plibs]

""" --------------------------------- tests -------------------------------- """
def test_moptimizer_scan_and_prefilter(MOptimizers, mock_input):
    optimizer = MOptimizers[0]
    match, pairs, flags = optimizer.scan_and_prefilter(mock_input)
    assert (match.loss_matrix >= 0).all(), 'loss matrix has negative values'
    assert len(pairs) == len(mock_input.qcluster_v) * len(mock_input.flash_v), 'pairs has wrong length'
    assert len(pairs) == len(flags), 'pairs and flags have different lengths'
    
def test_moptimizer_fit(MOptimizers, mock_input):
    optimizer_prefilter = MOptimizers[0]
    optimizer_no_prefilter = MOptimizers[2]
    
    for optimizer in [optimizer_prefilter, optimizer_no_prefilter]:
        match = optimizer.fit(mock_input)
        assert (match.loss_matrix >= 0).all(), 'loss matrix has negative values'
        assert match.loss_matrix.shape == (len(mock_input.qcluster_v), len(mock_input.flash_v)), 'loss matrix has wrong shape'

def test_moptimizer_prefilter(MOptimizers):
    mock_loss_matrix = torch.as_tensor([
        [0., 1., 2., 3.],
        [1., 0., 2., 4.],
        [2., 1., 0., 5.],
        [3., 2., 1., 0.]
    ])
    
    optimizer = MOptimizers[0]
    loss_threshold = 2.0
    top_k = 2
    
    result_row = optimizer.select_topk_2d(mock_loss_matrix, top_k, loss_threshold, mode='row')
    result_col = optimizer.select_topk_2d(mock_loss_matrix, top_k, loss_threshold, mode='col')

    expected_row = [[0,0],[0,1],
                [1,1],[1,0],
                [2,2],[2,1],
                [3,3],[3,2]]
    
    expected_col = [[0,0],[0,1],
                [1,0],[1,1],
                [2,2],[3,2],
                [3,3]]
    
    as_set = lambda x: set(map(tuple,x))  # noqa: E731
    assert as_set(result_row) == as_set(expected_row), 'row prefiltering failed'
    assert as_set(result_col) == as_set(expected_col), 'col prefiltering failed'

    result_both = optimizer.select_topk_2d(mock_loss_matrix, top_k, loss_threshold, mode='both')
    assert as_set(result_both) == as_set(result_row + result_col), 'both prefiltering failed'