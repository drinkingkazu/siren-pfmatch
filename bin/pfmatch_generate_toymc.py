#!/usr/bin/python
from pfmatch.utils import get_config, load_config, load_detector_config
from pfmatch.apps import ToyMC
from photonlib import PhotonLib, MultiLib
from pfmatch.io import H5File
import yaml
import fire
import tqdm

def main(cfg_file: str=None,
    cfg_keyword: str=None,
    num_events: int=10,
    plib_file: str='',
    mlib_file: str='',
    output_file: str='',
    num_tracks: int=0):

    if cfg_file is None and cfg_keyword is None:
        raise RuntimeError('Must provide either --cfg_file or --cfg_keyword flag.')
    if cfg_file and cfg_keyword:
        raise RuntimeError('Cannot specify both --cfg_file and --cfg_keyword flags.')

    if cfg_keyword:
        cfg_file = get_config(cfg_keyword)

    cfg = yaml.safe_load(open(cfg_file,'r'))
    det_cfg = load_detector_config(cfg)

    if plib_file:
        cfg['photonlib']=dict(filepath=plib_file)
    if mlib_file:
        cfg['multilib']=dict(filepath=mlib_file)
    if cfg.get('photonlib',dict()).get('filepath') is None and cfg.get('multilib',dict()).get('filepath') is None:
        raise RuntimeError('Must specify the photonlib or multilib filepath using a flag --plib_file or --mlib_file.')

    if cfg.get('multilib',dict()).get('filepath'):
        plib = MultiLib.load(cfg)
    else:
        plib = PhotonLib.load(cfg)

    if cfg.get('ToyMC') is None:
        raise RuntimeError('ToyMC configuration block is missing from the config file.')

    if num_tracks:
        assert isinstance(num_tracks,int), f'--num_tracks must be an integer (given {num_tracks})'
        assert num_tracks > 0, f'--num_tracks must be positive (given {num_tracks})'
        cfg['ToyMC']['NumTracks']=num_tracks

    gen = ToyMC(cfg,det_cfg,plib)

    if output_file:
        cfg['H5File']=dict(FileName=output_file,Mode='w')
    if cfg.get('H5File',dict()).get('FileName') is None:
        raise RuntimeError('Must specify the output file name using a flag --output_file or in the config file.')

    fout = H5File(cfg)

    for _ in tqdm.tqdm(range(num_events)):
        fmatch_input = gen.make_flashmatch_inputs()
        pts_ctr=[len(qc.qpt_v) for qc in fmatch_input.qcluster_v]
        pes_ctr=[len(f.pe_v)   for f  in fmatch_input.flash_v   ]
        if 0 in pts_ctr       : print('Empty qcluster found')
        elif 0 in pes_ctr     : print('Empty flash found'   )
        elif sum(pts_ctr) < 1 : print('No qcluster found'   )
        elif sum(pes_ctr) < 1 : print('No flash found'      )
        else: fout.write_one(fmatch_input.qcluster_v, fmatch_input.flash_v)
    
    fout.close()

if __name__ == '__main__':
    fire.Fire(main)