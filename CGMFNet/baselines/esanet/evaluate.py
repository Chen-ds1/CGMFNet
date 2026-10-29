from pathlib import Path
from runtime import read_config, evaluate

CONFIG = Path(__file__).with_name('config.json')

if __name__ == '__main__':
    evaluate(read_config(CONFIG))
