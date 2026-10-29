import json,itertools
import numpy as np
import config as C
from inference import Predictor
from data import read_lines
from geometry import match_lines,extract_lines
from row_postprocess import extract_rows

SOURCE = C.CHECKPOINT.parent
WORK = SOURCE/'postprocess_validation'

def summary(items,extractor):
    tp=fn=fp=0;errors=[];angles=[];counts=[]
    for key,p,gt in items:
        pred,raw=extractor(p)
        m,n,f=match_lines(gt,pred,C.MATCH_MAX_PIXEL_DISTANCE,C.MATCH_MAX_ANGLE_DEG,C.MIN_VERTICAL_OVERLAP)
        tp+=len(m);fn+=n;fp+=f
        errors.extend(v['lateral_px'] for v in m);angles.extend(v['angle_deg'] for v in m)
        counts.append({'sample_id':key,'gt':len(gt),'pred':len(pred),'tp':len(m),'fn':n,'fp':f})
    return {'tp':tp,'fn':fn,'fp':fp,'precision':tp/(tp+fp) if tp+fp else 0,
        'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
        'mean_lateral_px':float(np.mean(errors)) if errors else None,
        'mean_angle_deg':float(np.mean(angles)) if angles else None,'per_image':counts}

def main():
    WORK.mkdir(parents=True,exist_ok=True)
    meta=json.loads((SOURCE/'experiment.json').read_text())
    val=meta['splits']['val'];items=[]
    runner=None
    for key in val:
        cache=WORK/(key+'.npz')
        if not cache.exists():
            if runner is None:runner=Predictor(SOURCE/'best.pt')
            _,probs,_,_,_=runner.predict(C.DATA_ROOT/'rgb'/(key+'.png'),C.DATA_ROOT/'depths'/(key+'.png'))
            np.savez_compressed(cache,**probs)
        with np.load(cache) as z:p=z['line']
        gt=read_lines(C.DATA_ROOT/'row_lines'/(key+'.json'));items.append((key,p,gt))
    baseline=summary(items,lambda p:extract_lines(p,C.THRESHOLD,C.MIN_ROW_SPAN))
    trials=[]
    for threshold,support,radius in itertools.product([0.35,0.5,0.65],[0.15,0.25],[0.01,0.015]):
        params=dict(threshold=threshold,min_support=support,radius_fraction=radius)
        result=summary(items,lambda p:extract_rows(p,**params))
        trials.append(dict(parameters=params,metrics=result))
    best=max(trials,key=lambda t:(t['metrics']['f1'],-(t['metrics']['mean_lateral_px'] or 1e9)))
    report={'selection_split':'validation only','ids':val,'checkpoint':str(SOURCE/'best.pt'),
        'baseline':baseline,'selected':best,'trials':trials}
    (WORK/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (C.ROOT/'postprocess_config.json').write_text(json.dumps(best['parameters'],indent=2),encoding='utf-8')
    print('BASELINE',baseline);print('SELECTED',best)

if __name__=='__main__':main()
