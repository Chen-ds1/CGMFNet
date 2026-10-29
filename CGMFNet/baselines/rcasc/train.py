from rcasc.config import PROJECT_DIR, PREPARED_DIR
import argparse
from rcasc.engine import train

def main():
    parser = argparse.ArgumentParser(description='Train RCASC on a prepared dataset.')
    parser.add_argument('--prepared', default=str(PREPARED_DIR), help='Directory containing manifest.json and config.json')
    parser.add_argument('--config', help='Optional JSON configuration override')
    parser.add_argument('--data-root', help='Image directory after relocating the dataset')
    parser.add_argument('--output', default=str(PROJECT_DIR / 'runs' / 'full'))
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--resume', help='last.pt in the same output directory')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:0')
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('--batch-size must be positive')
    train(args)
if __name__ == '__main__':
    main()
