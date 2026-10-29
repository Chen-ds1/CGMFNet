from rcasc.config import PROJECT_DIR, PREPARED_DIR
import argparse
import json
from pathlib import Path
from rcasc.data import load_records
from rcasc.evaluation import evaluate
from rcasc.runtime import load_model

def main():
    parser = argparse.ArgumentParser(description='Evaluate a labeled dataset split.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prepared', default=str(PREPARED_DIR))
    parser.add_argument('--data-root')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--output', default=str(PROJECT_DIR / 'evaluation'))
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    model, config, device = load_model(args.checkpoint, args.device)
    records = [r for r in load_records(Path(args.prepared) / 'manifest.json', args.data_root) if r['split'] == args.split]
    if not records:
        parser.error('Selected subset is empty')
    print(json.dumps(evaluate(model, records, config, device, args.output), indent=2))
if __name__ == '__main__':
    main()
