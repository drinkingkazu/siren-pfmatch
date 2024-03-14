import pytest
import torch
import numpy as np

from pfmatch.apps import XOptimizer
from tests.fixtures import plib, siren, num_pmt, fake_flashmatch_data

@pytest.fixture
def XOptimizers(plib, siren):
    configs = [
        dict(),
        dict(XOptimizer=dict(LearningRateScheduler=dict(Name="ReduceLROnPlateau"))),
        dict(XOptimizer=dict(LearningRateScheduler=dict(Name="ReduceLROnPlateau", Parameters=dict(mode='min',
                                                                                                  factor=0.9,
                                                                                                  threshold=0.01,
                                                                                                  threshold_mode='rel'))))
    ]
    plibs = [plib, siren]
    return [XOptimizer(cfg, p) for cfg in configs for p in plibs]


""" --------------------------------- tests -------------------------------- """
def test_XOptimizer_init(XOptimizers):
    # this is tested by the fixture itself
    pass

def test_XOptimizer_fit(XOptimizers, fake_flashmatch_data, num_pmt):
    for algo in XOptimizers:
        qcluster_v, flash_v = fake_flashmatch_data[0]
        q = qcluster_v[0]
        f = flash_v[0]
        q.shift_to_center(algo.plib)

        dx_min, dx_max = algo.model.dx_range(q.qpt_v)
        dx_min += q.qpt_v[:,0].min()
        dx_max += q.qpt_v[:,0].max()

        result = algo.fit(q, f)['fit']
        loss, xmin, reco_pe, = result['loss_best'],result['xmin_best'],result['pe_best']
        assert loss >= 0, 'loss is negative'
        assert reco_pe.shape == (num_pmt,), 'reco_pe shape is not correct'
        assert torch.all(reco_pe>=0), 'reco_pe is negative'
        assert xmin >= dx_min and xmin <= dx_max, 'x_min is not within dx_min and dx_max'        
        
        # test XOptimizer properties after fit
        assert len(algo.hypothesis_history) == len(algo.loss_history), 'history length mismatch'
        assert np.allclose(loss, algo.loss)
        assert np.allclose(xmin, algo.xmin)
        assert torch.allclose(reco_pe, algo.hypothesis)
        assert algo.time_spent != 0, 'time_spent is zero'
        
        result = algo.fit(q, f, dx=1e8)['fit']
        loss, xmin, reco_pe, = result['loss_best'],result['xmin_best'],result['pe_best']
        assert loss >= 0, 'loss is negative'
        assert reco_pe.shape == (num_pmt,), 'reco_pe shape is not correct'
        assert torch.all(reco_pe>=0), 'reco_pe is negative'
        assert xmin >= dx_min and xmin <= dx_max, 'x_min is not within dx_min and dx_max'        


def test_XOptimizer_scan_loss(XOptimizers, fake_flashmatch_data, num_pmt):
    for algo in XOptimizers:
        qcluster_v, flash_v = fake_flashmatch_data[0]
        q = qcluster_v[0]
        f = flash_v[0]
        q.shift_to_center(algo.plib)
        
        result = algo.scan_loss(q.qpt_v, f.pe_v)
        loss, dx, pe = result['loss_init'],result['dx_init'],result['pe_init']
        assert loss >= 0, 'loss is negative'
        assert pe.shape == (num_pmt,), 'reco_pe shape is not correct'
        assert torch.all(pe>=0), 'reco_pe is negative'
        
        dx_min, dx_max = algo.model.dx_range(q.qpt_v)
        
        assert dx >= dx_min and dx <= dx_max, 'dx is not within dx_min and dx_max'