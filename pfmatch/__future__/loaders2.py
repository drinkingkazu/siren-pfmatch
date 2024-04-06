import numpy as np
import torch

def SimpleLoader(ds,batch_size,terminate=True,concat=False):
    
    start_index = 0
    all_idx = np.arange(len(ds))
    interrupt=False
    while not interrupt:
        
        last_index = start_index + batch_size
        
        if last_index <= len(ds):
            #print(start_index,'=>',last_index)
            idx_v = all_idx[start_index:last_index]
            start_index = last_index
            
        elif terminate:
            #print(start_index,'=> end')
            idx_v = all_idx[start_index:]
            interrupt = True
            
        else:
            #print(start_index,'=> end then 0 =>',(last_index-len(ds)))
            idx_v = np.concatenate([all_idx[start_index:],all_idx[:(last_index-len(ds))]])
            start_index = last_index - len(ds)
        
        res = ds.select(idx_v,concat)
        batch = dict(qclusters=res[0], flashes=res[1])
        if concat:
            batch['sizes'] = res[2]            
        yield batch

        
def RandomLoader(ds,batch_size,terminate=True,concat=False):
    
    start_index = 0
    all_idx = np.arange(len(ds))
    np.random.shuffle(all_idx)
    interrupt=False
    while not interrupt:
        
        last_index = start_index + batch_size
        
        if last_index <= len(ds):
            idx_v = all_idx[start_index:last_index]
            start_index = last_index
            
        elif terminate:
            idx_v = all_idx[start_index:]
            interrupt = True
            
        else:
            start_index = 0
            last_index = batch_size
            np.random.shuffle(all_idx)
            idx_v = all_idx[start_index:last_index]
            start_index = last_index
        
        res = ds.select(idx_v,concat)
        batch = dict(qclusters=res[0], flashes=res[1])
        if concat:
            batch['sizes'] = res[2]            
        yield batch
        

def RandomSequenceLoader(ds,batch_size,terminate=True,concat=False):
    
    interrupt = int(len(ds)/batch_size)
    while True:
        if terminate and interrupt==0:
            break

        start_index = int(np.random.uniform()*(len(ds)-batch_size))
        res = ds.slice(start_index,start_index+batch_size)        
        batch = dict(qclusters=res[0], flashes=res[1], sizes=res[2])
        if not concat:
            batch['qclusters']=torch.split(batch['qclusters'],batch['sizes'])
            batch.pop('sizes')          
        yield batch
        interrupt -= 1

