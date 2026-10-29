import argparse
from prsgnet.runtime import load_config,resolve,add_inference_arguments,choose_weights,choose_score,device_name,new_output,load_model
from prsgnet.evaluation import run,read_records


def main(argv=None):
    config,path = load_config(argv)
    p = argparse.ArgumentParser(description='Predict crop rows from images.')
    add_inference_arguments(p,config,path)
    inputs = p.add_mutually_exclusive_group()
    inputs.add_argument('--input',help='Image or directory; defaults to config inference.input')
    inputs.add_argument('--manifest',help='JSONL list; exports comparisons when rows are present')
    args = p.parse_args(argv)
    if args.manifest:
        source = resolve(args.manifest)
        records = read_records(source)
    else:
        source = resolve(args.input or config['inference']['input'])
        if not source.exists():
            raise FileNotFoundError(source)
        files = [source] if source.is_file() else sorted(f for f in source.iterdir() if f.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))
        records = [{'image':str(f)} for f in files]
    if not records:
        raise ValueError(f'No images: {source}')
    weights = choose_weights(args.weights)
    score = choose_score(args.score,weights)
    device = device_name(args.device)
    output = new_output('predict',args.out)
    run(load_model(weights,device),records,device,score,args.nms_distance,output,
        metadata={'task':'predict','weights':str(weights),'source':str(source)}, line_mode=args.line_mode, warmup=args.warmup)


if __name__ == '__main__':
    main()
