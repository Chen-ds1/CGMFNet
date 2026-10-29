from rcasc.config import INPUT_DIR, PREPARED_DIR
import argparse
from rcasc.config import DEFAULT
from rcasc.data import prepare
from rcasc.utils import load

def main():
    parser = argparse.ArgumentParser(description='Convert LabelMe annotations into an RCASC manifest.')
    parser.add_argument('--data', default=str(INPUT_DIR))
    parser.add_argument('--output', default=str(PREPARED_DIR))
    parser.add_argument('--config', help='Optional JSON configuration')
    parser.add_argument('--mode', choices=['all', 'core4'], default=None)
    args = parser.parse_args()
    config = DEFAULT.copy()
    if args.config:
        config.update(load(args.config))
    if args.mode:
        config['mode'] = args.mode
        config['max_rows'] = 9 if args.mode == 'all' else 4
    prepare(args.data, args.output, config)
if __name__ == '__main__':
    main()
