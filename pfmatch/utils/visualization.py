import plotly.graph_objects as go
from pfmatch.utils import *
import yaml
    
def get_trace_box(min_pt, max_pt):
    xmin,ymin,zmin = min_pt
    xmax,ymax,zmax = max_pt
    box = go.Mesh3d(
        # 8 vertices of a cube
        x=[xmin, xmin, xmax, xmax, xmin, xmin, xmax, xmax],
        y=[ymin, ymax, ymax, ymin, ymin, ymax, ymax, ymin],
        z=[zmin, zmin, zmin, zmin, zmax, zmax, zmax, zmax],

        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.1,
        color='blue',
        flatshading = True
    ) 

    return box

def get_trace_active_volume(keyword):
    cfg=yaml.safe_load(open(get_config(keyword),'r').read())
    min_pt = cfg.get('MinPosition')
    max_pt = cfg.get('MaxPosition')
    return get_trace_box(min_pt,max_pt)

def get_trace_tpc(keyword):
    cfg=yaml.safe_load(open(get_config(keyword),'r').read())
    ctr=0
    traces=[]
    while True:
        pos=cfg.get('TPC%d' % ctr)
        if pos is None:
            break
        traces.append(get_trace_box(*pos))
        ctr+=1
    return traces

def get_trace_pmt(keyword,values=None):
    cfg=yaml.safe_load(open(get_config(keyword),'r').read())
    ctr=0
    pts=[]
    while True:
        pos = cfg.get('PMT%d' % ctr)
        if pos is None:
            break
        pts.append(pos)
        ctr+=1
    pts = np.array(pts)
    trace = go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],
                         mode='markers',
                         marker=dict(size=4,
                                     color=None if values is None else values,
                                    ),
                        )
    return trace