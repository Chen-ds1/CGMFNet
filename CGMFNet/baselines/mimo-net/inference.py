import json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F
import config as C
from model import MIMONet
from data import input_arrays,tensor
from row_postprocess import extract_rows, DEFAULTS
from extend_rows import render_full

class Predictor:
    def __init__(self,checkpoint):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.checkpoint=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.cfg=self.checkpoint['model_config'];self.meta=self.checkpoint['metadata']
        self.model=MIMONet(**self.cfg,pretrained=False).to(self.device)
        self.model.load_state_dict(self.checkpoint['model'],strict=True);self.model.eval()
        params_path=C.ROOT/'postprocess_config.json'
        self.postprocess=dict(DEFAULTS)
        if params_path.exists():self.postprocess.update(json.loads(params_path.read_text(encoding='utf-8')))
    def sync(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
    def inputs(self,r,d):
        a,b=input_arrays(r,d,self.cfg['image_hw'],self.cfg['depth_mode'],self.meta['depth_scale_meters'])
        return tensor(a)[None],tensor(b)[None]
    @torch.inference_mode()
    def warmup(self,r,d):
        a,b=[t.to(self.device) for t in self.inputs(r,d)]
        for _ in range(C.WARMUP):self.model(a,b)
        self.sync()
    @torch.inference_mode()
    def predict(self,r,d):

        with Image.open(r) as im: original=im.convert('RGB').copy()
        with Image.open(d) as im: depth=im.copy()
        self.sync();t0=time.perf_counter()
        h,w=self.cfg['image_hw']
        rgb=np.asarray(original.resize((w,h),Image.Resampling.BILINEAR),np.float32)/255
        if self.cfg['depth_mode']=='pseudocolor':
            if depth.mode!='RGB':raise ValueError('Expected RGB pseudocolor depth')
            dep=np.asarray(depth.resize((w,h),Image.Resampling.BILINEAR),np.float32)/255
        else:
            dep=np.asarray(depth.resize((w,h),Image.Resampling.NEAREST),np.float32)
            if dep.ndim!=2:raise ValueError('Expected metric depth array')
            dep=np.clip(dep*self.meta['depth_scale_meters'],0,10)[...,None]/10
        a,b=tensor(rgb)[None],tensor(dep)[None];t1=time.perf_counter()
        a,b=a.to(self.device),b.to(self.device);self.sync();t2=time.perf_counter()
        out=self.model(a,b);self.sync();t3=time.perf_counter()
        probs={k:F.interpolate(v,size=(original.height,original.width),mode='bilinear',align_corners=False).sigmoid()[0,0].cpu().numpy() for k,v in out.items()}
        lines,raw=extract_rows(probs['line'],**self.postprocess)
        self.sync();t4=time.perf_counter()
        timing=dict(preprocess_ms=1000*(t1-t0),input_transfer_ms=1000*(t2-t1),inference_ms=1000*(t3-t2),
                    postprocess_ms=1000*(t4-t3),detection_total_ms=1000*(t4-t0))
        return original,probs,lines,raw,timing

def save_prediction(out,key,original,probs,lines,raw):
    out=Path(out)
    for folder in ['probabilities','plant_masks','line_masks','lines','overlays','overlays_segments','lines_full']:(out/folder).mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'probabilities'/f'{key}.npz',**probs)
    for name in ['plant','line']:Image.fromarray(((probs[name]>=C.THRESHOLD)*255).astype('uint8')).save(out/f'{name}_masks'/f'{key}.png')
    data={'lines':[p.tolist() for p in lines],'raw_tracks':[p.tolist() for p in raw],
          'coordinate_system':'original image pixels','width':original.width,'height':original.height}
    (out/'lines'/f'{key}.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    overlay=original.copy();draw=ImageDraw.Draw(overlay)
    for i,p in enumerate(lines):
        draw.line([tuple(v) for v in p],fill=(255,0,0),width=2)
        draw.text(tuple(p[-1]),str(i),fill=(255,255,0))
    overlay.save(out/'overlays_segments'/f'{key}.png')
    full,extended=render_full(original,lines,raw)
    full.save(out/'overlays'/f'{key}.png')
    (out/'lines_full'/f'{key}.json').write_text(json.dumps({'purpose':'boundary extension for visualization',
        'width':original.width,'height':original.height,'lines':extended},indent=2),encoding='utf-8')
