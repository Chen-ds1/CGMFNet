from pathlib import Path
import json, random, hashlib
import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.utils.data import Dataset

def read_lines(path):
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    lines=[]
    for s in d['shapes']:
        if s.get('label')!='rice_row': continue
        if s.get('shape_type') not in ('line','linestrip'): raise ValueError(f'Unexpected shape: {path}')
        p=np.asarray(s['points'],dtype=float)
        if len(p)<2 or not np.isfinite(p).all(): raise ValueError(f'Invalid line: {path}')
        lines.append(p)
    return lines

def line_mask(lines,original_wh,hw,width):
    h,w=hw; ow,oh=original_wh
    im=Image.new('L',(w,h)); draw=ImageDraw.Draw(im)
    for p in lines:
        xy=[(float(x*w/ow),float(y*h/oh)) for x,y in p]
        draw.line(xy,fill=255,width=width)
    return np.asarray(im,dtype=np.float32)/255

def input_arrays(rgb_path,depth_path,hw,mode,scale):
    h,w=hw
    with Image.open(rgb_path) as im: rgb=np.array(im.convert('RGB').resize((w,h),Image.Resampling.BILINEAR),dtype=np.float32)/255
    with Image.open(depth_path) as im:
        if mode=='pseudocolor':
            if im.mode!='RGB': raise ValueError('Expected RGB pseudocolor depth')
            dep=np.array(im.resize((w,h),Image.Resampling.BILINEAR),dtype=np.float32)/255
        elif mode=='metric':
            if len(np.asarray(im).shape)!=2: raise ValueError('Metric depth must be single-channel')
            dep=np.asarray(im.resize((w,h),Image.Resampling.NEAREST),dtype=np.float32)
            dep=np.clip(dep*scale,0,10)[...,None]/10
        else: raise ValueError(mode)
    return rgb,dep

def tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))).float()

class RiceDataset(Dataset):
    def __init__(self,root,ids,hw,line_width,depth_mode,scale,augment=False):
        self.root=Path(root); self.ids=ids; self.hw=tuple(hw); self.width=line_width
        self.mode=depth_mode; self.scale=scale; self.augment=augment
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        key=self.ids[i]; r=self.root
        rgb,dep=input_arrays(r/'rgb'/f'{key}.png',r/'depths'/f'{key}.png',self.hw,self.mode,self.scale)
        with Image.open(r/'plant_masks'/f'{key}.png') as im:
            wh=im.size
            plant=(np.asarray(im.resize(self.hw[::-1],Image.Resampling.NEAREST))>0).astype(np.float32)
        line=line_mask(read_lines(r/'row_lines'/f'{key}.json'),wh,self.hw,self.width)
        if self.augment and random.random()<0.5:
            rgb,dep,plant,line=[np.flip(a,axis=1).copy() for a in [rgb,dep,plant,line]]
        return tensor(rgb),tensor(dep),torch.from_numpy(plant[None]),torch.from_numpy(line[None]),key

def audit(root):
    root=Path(root); names={}
    for folder,ext in [('rgb','.png'),('depths','.png'),('plant_masks','.png'),('row_lines','.json')]:
        names[folder]={p.stem for p in (root/folder).glob('*'+ext)}
    ids=sorted(names['rgb'])
    if not ids or any(v!=set(ids) for v in names.values()): raise ValueError('Four folders must contain identical nonempty sample IDs')
    hashes={}; modes={}; foreground=[]
    for key in ids:
        with Image.open(root/'rgb'/f'{key}.png') as im:
            size=im.size; digest=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
        if digest in hashes: raise ValueError(f'Duplicate RGB content: {key}, {hashes[digest]}')
        hashes[digest]=key
        for folder in ['depths','plant_masks']:
            with Image.open(root/folder/f'{key}.png') as im:
                if im.size!=size: raise ValueError(f'Size mismatch: {key}')
                modes.setdefault(folder,set()).add(im.mode)
                if folder=='plant_masks':
                    a=np.asarray(im)
                    if a.ndim!=2 or not set(np.unique(a)).issubset({0,1,255}): raise ValueError(f'Not a binary mask: {key}')
                    foreground.append(float((a>0).mean()))
        d=json.loads((root/'row_lines'/f'{key}.json').read_text(encoding='utf-8'))
        if (d['imageWidth'],d['imageHeight'])!=size: raise ValueError(f'Label dimensions: {key}')
        for p in read_lines(root/'row_lines'/f'{key}.json'):
            if (p<0).any() or (p[:,0]>size[0]).any() or (p[:,1]>size[1]).any(): raise ValueError(f'Out-of-frame annotation: {key}')
    return ids,{'count':len(ids),'modes':{k:sorted(v) for k,v in modes.items()},'foreground_fraction_range':[min(foreground),max(foreground)]}

def validate_splits(s,ids):
    parts=[s[k] for k in ['train','val','test']]
    if any(not p or len(p)!=len(set(p)) for p in parts): raise ValueError('Empty/duplicate split')
    if sum(map(len,parts))!=len(set(sum(parts,[]))): raise ValueError('Split leakage')
    if set(sum(parts,[]))!=set(ids): raise ValueError('Splits must cover data exactly')
