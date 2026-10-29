import csv,json,time
from datetime import datetime
import numpy as np
import torch
from PIL import Image
from openpyxl import Workbook
import config as C
from data import audit,validate_splits,read_lines,line_mask
from inference import Predictor,save_prediction
from geometry import match_lines,segmentation_stats

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def stats(values):
    a=np.asarray([v for v in values if v is not None],float)
    return {'n':len(a),'mean':float(a.mean()) if len(a) else None,
        'sd':float(a.std(ddof=1)) if len(a)>1 else None,
        'median':float(np.median(a)) if len(a) else None,'p95':float(np.percentile(a,95)) if len(a) else None}

def main():
    ids,_=audit(C.DATA_ROOT)
    runner=Predictor(C.CHECKPOINT)
    splits=runner.meta['splits'];validate_splits(splits,ids)
    current=json.loads(C.SPLITS.read_text(encoding='utf-8'))
    if any(current[k]!=splits[k] for k in ['train','val','test']):raise ValueError('Current split differs from checkpoint split')
    test=splits['test'];out=C.RUN_ROOT/('evaluation_'+datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True,exist_ok=False)
    hs=json.loads(C.GROUND_HOMOGRAPHIES.read_text(encoding='utf-8')) if C.GROUND_HOMOGRAPHIES else None
    if hs is not None and any(k not in hs for k in test):raise ValueError('Missing per-image ground homography')
    root=C.DATA_ROOT
    runner.warmup(root/'rgb'/f'{test[0]}.png',root/'depths'/f'{test[0]}.png')
    images=[];pairs=[];timings=[]
    for key in test:
        original,probs,pred,raw,timing=runner.predict(root/'rgb'/f'{key}.png',root/'depths'/f'{key}.png')
        gt=read_lines(root/'row_lines'/f'{key}.json')
        matches,fn,fp=match_lines(gt,pred,C.MATCH_MAX_PIXEL_DISTANCE,C.MATCH_MAX_ANGLE_DEG,C.MIN_VERTICAL_OVERLAP,hs[key] if hs else None)
        row={'sample_id':key,'gt_rows':len(gt),'pred_rows':len(pred),'tp_rows':len(matches),'fn_rows':fn,'fp_rows':fp,
             'angle_deg_mean':stats([p['angle_deg'] for p in matches])['mean'],
             'lateral_px_mean':stats([p['lateral_px'] for p in matches])['mean']}
        with Image.open(root/'plant_masks'/f'{key}.png') as im:target=np.asarray(im)>0
        network_target=line_mask(gt,original.size,runner.cfg['image_hw'],runner.meta['line_width'])
        line_target=np.asarray(Image.fromarray((network_target*255).astype('uint8')).resize(original.size,Image.Resampling.NEAREST))>0
        targets={'plant':target,'line':line_target}
        for name in targets:
            row.update({name+'_'+k:v for k,v in segmentation_stats(probs[name]>=C.THRESHOLD,targets[name]).items()})
        if hs:
            row['lateral_cm_mean']=stats([p['lateral_cm'] for p in matches])['mean']
            row['ground_angle_deg_mean']=stats([p['ground_angle_deg'] for p in matches])['mean']
        images.append(row);pairs.extend(dict(sample_id=key,**p) for p in matches)
        save_start=time.perf_counter();save_prediction(out,key,original,probs,pred,raw)
        timing['save_ms']=(time.perf_counter()-save_start)*1000
        timings.append(dict(sample_id=key,**timing))
    tp=sum(r['tp_rows'] for r in images);fn=sum(r['fn_rows'] for r in images);fp=sum(r['fp_rows'] for r in images)
    summary={'image_count':len(test),'matched_rows':tp,'missed_rows':fn,'false_rows':fp,
        'row_precision':tp/(tp+fp) if tp+fp else None,'row_recall':tp/(tp+fn) if tp+fn else None,
        'row_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        'angle_deg_matched_pairs':stats([p['angle_deg'] for p in pairs]),
        'lateral_px_matched_pairs':stats([p['lateral_px'] for p in pairs]),
        'angle_deg_per_image':stats([p['angle_deg_mean'] for p in images]),
        'lateral_px_per_image':stats([p['lateral_px_mean'] for p in images])}
    if hs:
        summary['lateral_cm_matched_pairs']=stats([p['lateral_cm'] for p in pairs])
        summary['ground_angle_deg_matched_pairs']=stats([p['ground_angle_deg'] for p in pairs])
    for name in ['plant','line']:
        for k in ['iou','miou','precision','recall','dice']:summary[f'{name}_{k}_per_image']=stats([r[f'{name}_{k}'] for r in images])
    for field in ['preprocess_ms','input_transfer_ms','inference_ms','postprocess_ms','detection_total_ms','save_ms']:
        summary[field]=stats([r[field] for r in timings])
    summary['model_fps']=1000*len(test)/sum(r['inference_ms'] for r in timings)
    summary['pipeline_fps']=1000*len(test)/sum(r['detection_total_ms'] for r in timings)
    summary['parameters']=sum(p.numel() for p in runner.model.parameters())
    summary['parameter_memory_mb_fp32']=summary['parameters']*4/1e6
    protocol={'checkpoint':str(C.CHECKPOINT),'epoch':runner.checkpoint['epoch'],'device':str(runner.device),
        'gpu':torch.cuda.get_device_name(0) if runner.device.type=='cuda' else None,'precision':'FP32','batch_size':1,'warmup':C.WARMUP,
        'timing':'synchronized; pipeline includes preprocessing, transfer, forward, resize and line extraction; excludes file read, evaluation, rendering and saving',
        'angle':'undirected TLS orientation difference in original image pixels, degrees',
        'lateral_px':'mean absolute x difference at 100 common y samples; matched rows only',
        'matching':{'max_px':C.MATCH_MAX_PIXEL_DISTANCE,'max_angle_deg':C.MATCH_MAX_ANGLE_DEG,'min_gt_vertical_overlap':C.MIN_VERTICAL_OVERLAP},
        'mask_threshold':C.THRESHOLD,
        'row_postprocess':runner.postprocess,
        'row_postprocess_selection':'fixed on validation split; no fixed number of rows',
        'sd':'sample standard deviation (ddof=1), not independent training repeats',
        'cm':'unavailable without validated image-to-ground homographies' if hs is None else 'symmetric point-to-fitted-line ground distance',
        'warning':'adapted implementation; align splits and metric protocol with other baselines before manuscript comparison'}
    write_csv(out/'per_image_metrics.csv',images);write_csv(out/'matched_rows.csv',pairs);write_csv(out/'per_image_timing.csv',timings)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    wb=Workbook();wb.remove(wb.active)
    flat=[]
    for k,v in summary.items():flat.append({'metric':k,**v} if isinstance(v,dict) else {'metric':k,'value':v})
    for name,rows in [('Summary',flat),('PerImage',images),('MatchedRows',pairs),('Timing',timings)]:
        ws=wb.create_sheet(name);keys=list(dict.fromkeys(k for r in rows for k in r))
        if keys:
            ws.append(keys)
            for r in rows:ws.append([r.get(k) for k in keys])
        ws.freeze_panes='A2'
    wb.save(out/'evaluation.xlsx')
    print('Results:',out)

if __name__=='__main__':main()
