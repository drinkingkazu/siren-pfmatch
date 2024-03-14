
import pytest
import yaml
import numpy as np
import os
import torch

from pfmatch.apps import ToyMC
from tests.fixtures import GLOBAL_SEED, plib, siren, test_data_dir

def cfg(TimeAlgo, TrackAlgo, NumTracks, TruncateTPC, PosXVariation, PEVariation, GLOBAL_SEED):
    #print('GLOBAL_SEED',GLOBAL_SEED())
    cfg = f"""
    ToyMC:
        TimeAlgo: "{TimeAlgo}" # random, periodic, same
        TrackAlgo: "{TrackAlgo}" # random, top-bottom
        PeriodTPC: [-1300.,1300] # in micro-second, [-340-965(drift time),1698.4-340]
        PeriodPMT: [-1000., 1000.] # in micro-second, [-1150, 3450-1150]
        PEVariation: {PEVariation} # channel-to-channel (PMT-wise) variation in (fake) reco PE amount
        LightYieldVariation: 0.00  # light yield variation per point in QCluster_t
        PosXVariation: {PosXVariation} # total x position variation in cm
        TruncateTPC: {TruncateTPC} # 1=truncate TPC trajectory mimicing readout effect
        NumTracks: "{NumTracks}"
        NumpySeed: {GLOBAL_SEED}
        NeutrinoTime: 0.
    """
    return dict(yaml.safe_load(cfg))

@pytest.fixture
def cfgs(GLOBAL_SEED):
    # note: assumes one change in 1 variable shouldn't affect functions
    #       using the other variables
    TimeAlgo = ["random", "periodic", "same"]
    TrackAlgo = ["random", "top-bottom"]
    NumTracks = [5, "6-10"]
    TruncateTPC = [0, 1]
    PosXVariation = [0.0, 10.0]
    PEVariation = [0.0, 100.0]
    
    max_len = max(len(TimeAlgo), len(TrackAlgo), len(NumTracks), \
                  len(TruncateTPC), len(PosXVariation), len(PEVariation))
    possible_cfgs = []
    for i in range(max_len):
        possible_cfgs.append(cfg(TimeAlgo[i%len(TimeAlgo)],
                                TrackAlgo[i%len(TrackAlgo)],
                                NumTracks[i%len(NumTracks)],
                                TruncateTPC[i%len(TruncateTPC)],
                                PosXVariation[i%len(PosXVariation)],
                                PEVariation[i%len(PEVariation)],
                                GLOBAL_SEED,
                                ),
                            )
    return possible_cfgs
    
@pytest.fixture
def det_cfg(test_data_dir):
    return yaml.load(open(os.path.join(test_data_dir, 'icarus_detector.yaml')), Loader=yaml.Loader)

@pytest.fixture
def ToyMCs(cfgs, det_cfg, plib, siren):
    return [ToyMC(cfg, det_cfg, lib) for cfg in cfgs for lib in [plib, siren]]

def test_gen_trajectories(rng, ToyMCs, det_cfg):
    for tmc in ToyMCs:
        num_tracks = rng.integers(2, 100)
        trajectories = np.asarray(tmc.gen_trajectories(num_tracks))
        assert trajectories.shape == (num_tracks, 2, 3), 'trajectories shape is not correct'
        mins = det_cfg['ActiveVolumeMin']
        maxs = det_cfg['ActiveVolumeMax']
        trajectories = trajectories.reshape(-1, 3)
        
        def assert_within_edge(trajectories, i):
            assert trajectories[:,i].min() >= mins[i], \
                f"trajectories in {'xyz'[i]} min out of bounds, {trajectories[:,i].min():.0f} < {mins[i]:.0f}"
            assert trajectories[:,i].max() <= maxs[i], \
                f"trajectories in {'xyz'[i]} max out of bounds, {trajectories[:,i].max():.0f} > {maxs[i]:.0f}"
        assert_within_edge(trajectories, 0)
        assert_within_edge(trajectories, 2)
        
        if tmc.track_algo == 'random':
            assert_within_edge(trajectories, 1)
        elif tmc.track_algo == 'top-bottom':
            assert np.allclose(trajectories[:,1].min(), mins[1]), \
                f"trajectories in y min out of bounds, {trajectories[:,1].min():.0f} < {mins[1]:.0f}"
            assert np.allclose(trajectories[:,1].max(), maxs[1]), \
                f"trajectories in y max out of bounds, {trajectories[:,1].max():.0f} > {maxs[1]:.0f}"

def test_gen_xt_shift(rng, ToyMCs):
    for tmc in ToyMCs:
        n_shifts = rng.integers(2, 100)
        time_dx_v = np.asarray(tmc.gen_xt_shift(n_shifts))
        
        ts, dxs = time_dx_v.T
        assert time_dx_v.shape == (n_shifts, 2), 'time_dx_v shape is not correct'
        assert (ts >= tmc.periodPMT[0]).all() and  (ts <= tmc.periodPMT[1]).all(), 'time_dx_v t out of bounds'

@pytest.fixture
def fake_tracks(rng, det_cfg):
    track = rng.random(size=(2,3))
    mins, maxs = det_cfg['ActiveVolumeMin'], det_cfg['ActiveVolumeMax']
    for i in range(3):
        track[:, i] = track[:,i]*(maxs[i]-mins[i]) + mins[i]
    return track

def test_make_qpoints(ToyMCs, fake_tracks, det_cfg):
    for tmc in ToyMCs:
        xmin = det_cfg['ActiveVolumeMin'][0]
        xmax = det_cfg['ActiveVolumeMax'][0]
        
        qpt_v = tmc.make_qpoints(fake_tracks)
        assert qpt_v.shape[1] == 4, 'qpt_v shape is not correct'
        assert torch.all(qpt_v[:,-1] >= 0), 'negative pe found in qpt_v. either track_to_qpoints' \
                                            'or applying light yield variation is broken'
        # 2024-03-14 Kazu - the current toymc.py implementation allows track to go beyond the boundary.
        #assert qpt_v[:,0].min() >= xmin, 'qpt_v x min out of bounds'
        #assert qpt_v[:,0].max() <= xmax, 'qpt_v x max out of bounds'
        

def test_make_photons(ToyMCs, fake_flashmatch_data, num_pmt):
    qc, __ = fake_flashmatch_data[0]
    qpt_v = qc[0].qpt_v
    for tmc in ToyMCs:    
        pe, pe_err, pe_true = tmc.make_photons(qpt_v)
        assert pe.shape == (num_pmt,), 'pe shape is not correct'
        assert pe_err.shape == (num_pmt,), 'pe_err shape is not correct'
        assert pe_true.shape == (num_pmt,), 'pe_true shape is not correct'
        assert torch.all(pe >= 0), 'negative pe found in pe'
        assert torch.all(pe_err >= 0), 'negative pe found in pe_err'
        assert torch.all(pe_true >= 0), 'negative pe found in pe_true'
    
def test_make_flashmatch_inputs(ToyMCs):
    for tmc in ToyMCs:
        # num_match = 10
        data = tmc.make_flashmatch_inputs(10)
        assert len(data.flash_v) <= 10, 'flash_v length is not correct'
        assert len(data.qcluster_v) <= 10, 'qcluster_v length is not correct'
        assert len(data.raw_qcluster_v) == len(data.qcluster_v), 'raw_qcluster_v length is not correct'
        assert all(hasattr(qc, 'xmin_true') for qc in data.qcluster_v), 'qclusters do not have truth info'
        assert all(hasattr(fl, 'time_true') for fl in data.flash_v), 'flashes do not have truth info'
        
        expected_match_pairs = np.tile(np.arange(min(len(data.flash_v),len(data.qcluster_v)))[:, None], 2)
        expected_match_pairs = list(map(tuple, expected_match_pairs))
        assert len(data.true_match) == min(len(data.flash_v),len(data.qcluster_v)), 'true_match length is not correct'

        # num_match = config
        data = tmc.make_flashmatch_inputs()
        assert len(data.flash_v) >= tmc.num_tracks[0] and len(data.flash_v) <= tmc.num_tracks[1], \
            'number of flashmatch pairs created does not match number specified in config'
        