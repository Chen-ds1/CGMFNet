import json,random
from PIL import Image
import config as C
from data import audit,validate_splits,read_lines

def main():
    ids,report=audit(C.DATA_ROOT)
    if C.SPLITS.exists():
        s=json.loads(C.SPLITS.read_text(encoding='utf-8'))
    else:

        groups={}
        for key in ids: groups.setdefault(key.split('_')[0],[]).append(key)
        keys=sorted(groups); random.Random(C.SEED).shuffle(keys)
        if len(keys)<3:
            raise ValueError('Fewer than three capture dates. Supply splits.json by independent field/sequence; no silent random-frame split.')
        n=max(1,round(len(keys)*0.2)); test=keys[:n]; val=keys[n:2*n]; train=keys[2*n:]
        s={k:sorted(sum((groups[g] for g in gs),[])) for k,gs in [('train',train),('val',val),('test',test)]}
        s['split_policy']='capture-date groups; provisional until aligned with existing baselines'
        validate_splits(s,ids)
        C.SPLITS.write_text(json.dumps(s,indent=2,ensure_ascii=False),encoding='utf-8')
    validate_splits(s,ids)
    out=C.RUN_ROOT/'data_audit'; out.mkdir(parents=True,exist_ok=True)
    report['splits']={k:len(s[k]) for k in ['train','val','test']}
    report['split_policy']=s.get('split_policy','user supplied')
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    for key in ids[:3]:
        im=Image.open(C.DATA_ROOT/'rgb'/f'{key}.png').convert('RGB')
        mask=Image.open(C.DATA_ROOT/'plant_masks'/f'{key}.png').convert('L')
        color=Image.new('RGB',im.size,(0,255,0)); overlay=Image.composite(Image.blend(im,color,0.4),im,mask)
        from PIL import ImageDraw
        draw=ImageDraw.Draw(overlay)
        for p in read_lines(C.DATA_ROOT/'row_lines'/f'{key}.json'): draw.line([tuple(v) for v in p],fill=(255,0,0),width=2)
        overlay.save(out/f'{key}.png')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
