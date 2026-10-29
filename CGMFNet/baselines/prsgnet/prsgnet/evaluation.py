import csv
import html
import json
import time
import platform
from pathlib import Path
import cv2
import numpy as np
import torch
from prsgnet.data import image_tensor, encode_rows, resolve_record_image
from prsgnet.postprocess import decode, counts
from prsgnet.line_geometry import extend_line


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def save_image(path, image):
    path.parent.mkdir(parents=True,exist_ok=True)
    ok, encoded = cv2.imencode('.png',image)
    if not ok:
        raise RuntimeError(f'Failed to encode: {path}')
    encoded.tofile(str(path))


def read_records(manifest):
    records = [json.loads(s) for s in manifest.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if not records:
        raise ValueError(f'Empty manifest: {manifest}')
    for rec in records:
        rec['image'] = str(resolve_record_image(rec['image'], manifest))
    return records


def forward(model, image, device):
    with torch.inference_mode():
        result = model(image_tensor(cv2.resize(image,(800,320)))[None].to(device))['stages'][-1]
    return {k:v[0] for k,v in result.items()}


def ground_truth(record, width, height):
    resized = [np.asarray(row)*[799/(width-1),319/(height-1)] for row in record['rows']]
    target = encode_rows(resized,800,320)
    if len(target['xs']) != len(record['rows']):
        raise ValueError(f'Some rows cannot be encoded: {record["image"]}')
    yy = np.linspace(319,0,36)
    return [np.c_[x[v].numpy()*799,yy[v.numpy()]].tolist() for x,v in zip(target['xs'],target['valid'])]


def metrics(values):
    tp,fp,fn = map(int,values)
    return {'TP':tp,'FP':fp,'FN':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'F1':2*tp/max(1,2*tp+fp+fn)}


def prepare_coordinates(rows, width, height, line_mode='extended'):
    items = []
    for row in rows:
        points = [[x*(width-1)/799,y*(height-1)/319] for x,y in row['points']]
        extended = extend_line(points,width,height)
        displayed = extended if line_mode == 'extended' and extended else points
        items.append({'score':row['score'],'points':points,'extended_points':extended,
                      'displayed_points':displayed,'extension_valid':bool(extended)})
    return items


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def timed_detection(model, image, device, score, nms_distance, line_mode):
    synchronize(device)
    with torch.inference_mode():
        t0 = time.perf_counter()
        tensor = image_tensor(cv2.resize(image,(800,320)))[None]
        t1 = time.perf_counter()
        tensor = tensor.to(device)
        synchronize(device)
        t2 = time.perf_counter()
        stage = model(tensor)['stages'][-1]
        synchronize(device)
        t3 = time.perf_counter()
        rows = decode({k:v[0] for k,v in stage.items()},320,800,score,nms_distance)
        items = prepare_coordinates(rows,image.shape[1],image.shape[0],line_mode)
        synchronize(device)
        t4 = time.perf_counter()
    timings = {'preprocess_ms':(t1-t0)*1000, 'input_transfer_ms':(t2-t1)*1000,
               'inference_ms':(t3-t2)*1000, 'postprocess_ms':(t4-t3)*1000,
               'detection_total_ms':(t4-t0)*1000, 'equivalent_fps':1/max(t4-t0,1e-12)}
    return rows,items,timings


def export_sample(out, index, record, image, rows, line_mode='extended', items=None):
    height,width = image.shape[:2]
    prediction = image.copy()
    raw_prediction = image.copy()
    if items is None:
        items = prepare_coordinates(rows,width,height,line_mode)
    for item in items:
        cv2.polylines(raw_prediction,[np.rint(item['points']).astype(np.int32)],False,(0,0,255),2)
        cv2.polylines(prediction,[np.rint(item['displayed_points']).astype(np.int32)],False,(0,0,255),2)
    stem = f'{index+1:04d}_{Path(record["image"]).stem}'
    image_file = out/'predictions'/(stem+'.png')
    save_image(image_file,prediction)
    raw_file = out/'predictions_raw'/(stem+'.png')
    save_image(raw_file,raw_prediction)
    coordinate_file = out/'coordinates'/(stem+'.json')
    coordinate_file.parent.mkdir(parents=True,exist_ok=True)
    coordinate_file.write_text(json.dumps({'image':record['image'],'line_mode':line_mode,'rows':items},indent=2,ensure_ascii=False),encoding='utf-8')
    comparison_file = None
    if 'rows' in record:
        annotation = image.copy()
        for row in record['rows']:
            cv2.polylines(annotation,[np.rint(row).astype(np.int32)],False,(0,220,0),2)
        band = np.full((40,width*2,3),25,np.uint8)
        cv2.putText(band,f'MANUAL LABELS: {len(record["rows"])}',(12,27),cv2.FONT_HERSHEY_SIMPLEX,.6,(70,240,70),1)
        cv2.putText(band,f'PREDICTIONS ({line_mode}): {len(rows)}',(width+12,27),cv2.FONT_HERSHEY_SIMPLEX,.6,(70,70,255),1)
        comparison_file = out/'comparisons'/(stem+'.png')
        save_image(comparison_file,np.concatenate((band,np.concatenate((annotation,prediction),axis=1)),axis=0))
    return {'image':record['image'],'prediction_count':len(rows),
            'prediction':image_file.relative_to(out).as_posix(),
            'raw_prediction':raw_file.relative_to(out).as_posix(),
            'comparison':comparison_file.relative_to(out).as_posix() if comparison_file else None}


def run(model, records, device, score, nms_distance, out, evaluate=False, metadata=None, line_mode='extended', warmup=50):
    if line_mode not in ('raw','extended'):
        raise ValueError('line_mode must be raw or extended')
    if warmup < 0 or not records:
        raise ValueError('warmup must be nonnegative and records must not be empty')
    model.eval()
    if warmup:
        sample = read_image(records[0]['image'])
        for _ in range(warmup):
            timed_detection(model,sample,device,score,nms_distance,line_mode)
    totals = {t:np.zeros(3,dtype=int) for t in (.5,.75)}
    entries, per_image, timing_rows = [], [], []
    for index,record in enumerate(records):
        image = read_image(record['image'])
        rows,items,timing = timed_detection(model,image,device,score,nms_distance,line_mode)
        timing_rows.append({'index':index+1,'image_name':Path(record['image']).name,'image_path':record['image'],
                            'original_width':image.shape[1],'original_height':image.shape[0],
                            'predicted_rows':len(rows),**timing})
        entries.append(export_sample(out,index,record,image,rows,line_mode,items=items))
        if evaluate:
            truth = ground_truth(record,image.shape[1],image.shape[0])
            for threshold in totals:
                value = counts([r['points'] for r in rows],truth,320,800,threshold)
                totals[threshold] += value
                per_image.append({'image':record['image'],'iou_threshold':threshold,**metrics(value)})
    with (out/'timing_per_image.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file,fieldnames=list(timing_rows[0]))
        writer.writeheader(); writer.writerows(timing_rows)
    elapsed = np.array([r['detection_total_ms'] for r in timing_rows])
    inference = np.array([r['inference_ms'] for r in timing_rows])
    timing_summary = {'image_count':len(records),'warmup_iterations':warmup,'batch_size':1,
        'device':str(device),'device_name':torch.cuda.get_device_name(device) if torch.device(device).type=='cuda' else platform.processor(),
        'torch_version':str(torch.__version__),'input_shape':[1,3,320,800],'precision':'float32 (no autocast)',
        'cpu_threads':torch.get_num_threads(),'line_mode':line_mode,
        'mean_detection_ms':float(elapsed.mean()),'median_detection_ms':float(np.median(elapsed)),
        'p95_detection_ms':float(np.percentile(elapsed,95)), 'total_detection_ms':float(elapsed.sum()),
        'detection_fps':float(1000*len(records)/elapsed.sum()),
        'mean_inference_ms':float(inference.mean()),'inference_fps':float(1000*len(records)/inference.sum()),
        'timing_method':'perf_counter wall clock, CUDA synchronize at stage boundaries',
        'included':'CPU resize/normalization, input transfer, network forward, decoding/NMS, output transfer, original-size coordinates, straight-line fit/extension',
        'excluded':'model initialization, warmup, image read/decode from disk, drawing, file writing, metric calculation',
        'caveat':'one measured pass per image; sequential export between images may affect GPU power state; not continuous-stream throughput'}
    (out/'timing_summary.json').write_text(json.dumps(timing_summary,indent=2,ensure_ascii=False),encoding='utf-8')
    with (out/'timing_summary.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file,fieldnames=list(timing_summary))
        writer.writeheader(); writer.writerow(timing_summary)
    report = {**(metadata or {}),'score':score,'nms_distance':nms_distance,'image_count':len(records),'images':entries,
              'line_mode':line_mode,'metric_geometry':'raw network polylines, no fitting or extension','timing':timing_summary}
    if evaluate:
        report['metrics'] = {str(t):metrics(v) for t,v in totals.items()}
        (out/'metrics.json').write_text(json.dumps(report['metrics'],indent=2),encoding='utf-8')
        with (out/'per_image.csv').open('w',newline='',encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file,fieldnames=list(per_image[0]))
            writer.writeheader(); writer.writerows(per_image)
        print(json.dumps(report['metrics'],indent=2))
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    cards = []
    for e in entries:
        visual = e['comparison'] or e['prediction']
        cards.append(f'<figure><figcaption>{html.escape(Path(e["image"]).name)} | {e["prediction_count"]} predictions</figcaption>'
                     f'<a href="{e["prediction"]}">Displayed prediction</a> · <a href="{e["raw_prediction"]}">Raw prediction</a><img loading="lazy" src="{visual}"></figure>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>PRSGNet Predictions</title>'
    page += '<style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:0 20px;background:#f4f6f8}img{width:100%;display:block}figure{background:white;padding:14px;margin:22px 0}figcaption{margin-bottom:10px}</style>'
    description = 'Red lines are fitted and extended to image boundaries. Extrapolated segments are display geometry. ' if line_mode == 'extended' else 'Raw predicted polylines are displayed without fitting or extension. '
    page += f'<h1>PRSGNet Predictions</h1><p>{len(records)} images; confidence threshold {score:g}. {description}Green lines on the left of comparison images are manual annotations. Metrics use raw predictions.</p>'
    page += ''.join(cards)+'</html>'
    (out/'index.html').write_text(page,encoding='utf-8')
    print(f'Results: {out}',flush=True)
    return report
