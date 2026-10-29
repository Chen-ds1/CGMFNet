from pathlib import Path
from runtime import read_config, train

CONFIG = Path(__file__).with_name('config.json')

if __name__ == '__main__':
    train(read_config(CONFIG))
