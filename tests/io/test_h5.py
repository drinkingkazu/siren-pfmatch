from tempfile import NamedTemporaryFile
from typing import List
import numpy as np
import os
import pytest

from pfmatch.io.h5file import H5File
from pfmatch.datatypes import Flash, QCluster
from tests.fixtures import fake_flashmatch_data

""" -------------------------------- helpers ------------------------------- """

@pytest.fixture
def writable_temp_file():
    tf = NamedTemporaryFile('w', suffix='.h5', delete=False).name
    yield tf
    os.remove(tf)
    
def compare_len_sum(write: List[QCluster | Flash], read: List[QCluster | Flash]):
    clstype = write[0].__class__.__name__
    assert len(write) == len(read), f'{clstype} written vs read length mismatch ({len(write)} != {len(read)})'
    attributes = {'QCluster': 'qpt_v', 'Flash': 'pe_v'}
    
    if isinstance(write[0], (QCluster, Flash)):
        sumfunc = lambda x: getattr(x, attributes[clstype]).cpu().numpy().sum()
    else: # numpy array
        sumfunc = lambda x: x.sum()
    
    w_sum  = np.sum([sumfunc(f) for f in write])
    r_sum = np.sum([sumfunc(f) for f in read])

    assert abs(w_sum-r_sum) < max(abs(w_sum/1.e5),abs(r_sum/1.e5)), \
            f'{clstype} value sum mismatch ({w_sum} != {r_sum})'

""" --------------------------------- tests -------------------------------- """

def test_h5_write_read_one(fake_flashmatch_data, writable_temp_file):
    (qcluster_v,flash_v) = fake_flashmatch_data[0]

    f = H5File.open(writable_temp_file,'w')
    f.write_one(qcluster_v,flash_v)
    
    qcluster_vout, flash_vout = f.read_one(0)
    compare_len_sum(qcluster_v,qcluster_vout)
    compare_len_sum(flash_v,flash_vout)

    f.close()

def test_h5_write_mode(fake_flashmatch_data, writable_temp_file):
    (qcluster_v,flash_v) = fake_flashmatch_data[0]

    tmp_name = writable_temp_file
    f = H5File.open(tmp_name,'w')
    f.close()
    f = H5File.open(tmp_name,'r')
    with pytest.raises(ValueError):
        f.write_one(qcluster_v,flash_v)

    f.close()
    
def test_h5_write_read_many(fake_flashmatch_data, writable_temp_file):

    f = H5File.open(writable_temp_file,'w')
    qcluster_vv = []
    pe_vv = []
    for (qcluster_v,flash_v) in fake_flashmatch_data:
        qcluster_vv.append(qcluster_v)
        pe_vv.append(flash_v)
    f.write_many(qcluster_vv,pe_vv)
    f.close()
    
    f = H5File.open(writable_temp_file,'r')
    qcluster_vv_read, pe_vv_read = f.read_many(range(len(f)))
    for i in range(len(qcluster_vv)):
        qcluster_v = qcluster_vv[i]
        qcluster_v_read = qcluster_vv_read[i]
        pe_v = pe_vv[i]
        pe_v_read = pe_vv_read[i]
        compare_len_sum(qcluster_v,qcluster_v_read)
        compare_len_sum(pe_v,pe_v_read)

def test_h5_write_shape_mismatch(fake_flashmatch_data, writable_temp_file):

    f = H5File.open(writable_temp_file,'w')
    qcluster_vv = []
    pe_vv = []
    for i, (qcluster_v,flash_v) in enumerate(fake_flashmatch_data):
        qcluster_vv.append(qcluster_v)
        if i%2==0:
            pe_vv.append(flash_v)
    with pytest.raises(ValueError):
        f.write_many(qcluster_vv,pe_vv)
    f.close()

def test_h5_read_index(fake_flashmatch_data, writable_temp_file):

    f = H5File.open(writable_temp_file,'w')
    qcluster_vv = []
    pe_vv = []
    for (qcluster_v,flash_v) in fake_flashmatch_data:
        qcluster_vv.append(qcluster_v)
        pe_vv.append(flash_v)
    f.write_many(qcluster_vv,pe_vv)
    f.close()

    f = H5File.open(writable_temp_file,'r')
    qc_read, fl_read = f[(len(f)//2)]
    qc_write, fl_write = qcluster_vv[len(f)//2], pe_vv[len(f)//2]
    for qc_r, qc_w, fl_read, fl_write in zip(qc_read,qc_write, fl_read, fl_write):
        assert np.allclose(qc_r.qpt_v.cpu().numpy(),qc_w.qpt_v.cpu().numpy())
        assert np.allclose(fl_read.pe_v.cpu().numpy(),fl_write.pe_v.cpu().numpy())
    
    with pytest.raises(IndexError):
        f.read_one(len(f))
    with pytest.raises(IndexError):
        f.read_many([len(f)])
    f.close()
    
def test_h5_read_after_close(fake_flashmatch_data, writable_temp_file):
    f = H5File.open(writable_temp_file,'w')
    qcluster_vv = []
    pe_vv = []
    for (qcluster_v,flash_v) in fake_flashmatch_data:
        qcluster_vv.append(qcluster_v)
        pe_vv.append(flash_v)
    f.write_many(qcluster_vv,pe_vv)
    f.close()
    
    f = H5File.open(writable_temp_file,'r')
    f.close()
    with pytest.raises(ValueError):
        f.read_one(0)
    f.close()
    
def test_h5_read_blank(fake_flashmatch_data, writable_temp_file):
    f = H5File.open(writable_temp_file,'w')
    assert f.read_many([]) == ([], [])
    f.close()