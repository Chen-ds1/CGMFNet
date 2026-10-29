import argparse
from prsgnet.runtime import load_config,resolve,add_inference_arguments,choose_weights,choose_score,device_name,new_output,load_model
from prsgnet.evaluation import run,read_records


def main(argv=None, default_split='test'):
    config,path = load_config(argv)
    p = argparse.ArgumentParser(description='Evaluate a labeled validation or test split.')
    add_inference_arguments(p,config,path)
    p.add_argument('--split',choices=['val','test'],default=default_split)
    p.add_argument('--data',help='Override JSONL manifest; otherwise use configured manifest_dir/split.jsonl')
    args = p.parse_args(argv)
    manifest = resolve(args.data or str(resolve(config['data']['manifest_dir'])/(args.split+'.jsonl')))
    records = read_records(manifest)
    if any('rows' not in r for r in records):
        raise ValueError('Testing requires row annotations. Use predict.py for unlabeled images.')
    weights = choose_weights(args.weights)
    score = choose_score(args.score,weights)
    device = device_name(args.device)
    output = new_output(args.split,args.out)
    run(load_model(weights,device),records,device,score,args.nms_distance,output,evaluate=True,
        metadata={'task':args.split,'weights':str(weights),'manifest':str(manifest)}, line_mode=args.line_mode, warmup=args.warmup)


if __name__ == '__main__':
    main()
