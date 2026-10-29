import torch
from .model import RCASC

def load_model(checkpoint, device='auto'):
    torch.set_num_threads(4)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = RCASC(config).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return (model, config, device)
