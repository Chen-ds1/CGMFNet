import torch


@torch.no_grad()
def calibrate_bn(model, loader, device):
    modes = {module: module.training for module in model.modules()}
    momenta = {}
    model.eval()
    try:
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                momenta[module] = module.momentum
                module.reset_running_stats()
                module.momentum = None
                module.train()
        for images, _, _ in loader:
            model(images.to(device))
    finally:
        for module, momentum in momenta.items():
            module.momentum = momentum
        for module, mode in modes.items():
            module.training = mode
