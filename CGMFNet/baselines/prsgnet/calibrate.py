import argparse
import json
import numpy as np
from prsgnet.runtime import load_config,resolve,add_inference_arguments,choose_weights,device_name,load_model
from prsgnet.evaluation import read_records,read_image,forward,ground_truth,metrics
from prsgnet.postprocess import decode,counts


def main(argv=None):
    config,path = load_config(argv)
    p = argparse.ArgumentParser(description='Select a confidence threshold on the validation split.')
    add_inference_arguments(p,config,path)
    p.add_argument('--val',help='Validation JSONL; default configured data/labelme/val.jsonl')
    args = p.parse_args(argv)
    manifest = resolve(args.val or str(resolve(config['data']['manifest_dir'])/'val.jsonl'))
    records = read_records(manifest)
    weights = choose_weights(args.weights)
    device = device_name(args.device)
    model = load_model(weights,device)
    thresholds = [.1,.15,.2,.25,.3,.35,.4,.45,.5,.6]
    totals = {t:np.zeros(3,dtype=int) for t in thresholds}
    for record in records:
        image = read_image(record['image'])
        stage = forward(model,image,device)
        truth = ground_truth(record,image.shape[1],image.shape[0])
        for threshold in thresholds:
            rows = decode(stage,320,800,threshold,args.nms_distance)
            totals[threshold] += counts([r['points'] for r in rows],truth,320,800,.5)
    sweep = [{'score':t,**metrics(v),'F1_50':metrics(v)['F1']} for t,v in totals.items()]
    selected = max(sweep,key=lambda r:(r['F1_50'],r['score']))
    result = {'weights':str(weights),'validation_manifest':str(manifest),'selection_metric':'validation F1_50',
              'selected':selected,'sweep':sweep}
    out = resolve(args.out) if args.out else weights.parent/'calibration.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'Saved calibration: {out}')


if __name__ == '__main__':
    main()
