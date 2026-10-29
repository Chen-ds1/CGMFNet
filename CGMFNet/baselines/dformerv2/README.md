# DFormerv2 Rice Segmentation Baseline

A DFormerv2-S adaptation for binary rice segmentation, trained from random initialization using RGB and three-channel pseudocolor depth images. 

## Usage

Set the data paths, device, and training parameters in `config.json`, then run:

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

The scripts can also run directly in PyCharm. Before evaluation, set `checkpoint` to the trained `best.pth`. Results are saved under `runs/`. Use new output names in the configuration for repeated runs.

## Third-Party Sources and Licenses

- `dformerv2_encoder.py` is based on [DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation](https://github.com/VCIP-RGBD/DFormer), authored by yinbow, bowenyin@mail.nankai.edu.cn. The full MIT license is provided in `DFORMER_LICENSE`.
- `context_modules.py`, `model.py`, `model_utils.py`, `resnet.py`, and `rgb_depth_fusion.py` under `official_esanet/src/models/` come from [ESANet], authored by Mona Koehler, mona.koehler@tu-ilmenau.de, and Daniel Seichter, daniel.seichter@tu-ilmenau.de. The full Apache 2.0 license is provided in `official_esanet/LICENSE`.
- Parts of `context_modules.py` are adapted from [PSPNet]. Parts of `resnet.py` are adapted from [torchvision ResNet] and [SwiftNet].

The code includes local adaptations. Retain the original copyright, license, and modification notices.

[ESANet]: https://github.com/TUI-NICR/ESANet
[PSPNet]: https://github.com/hszhao/semseg/blob/master/model/pspnet.py
[torchvision ResNet]: https://github.com/pytorch/vision/blob/v0.4.2/torchvision/models/resnet.py
[SwiftNet]: https://github.com/orsic/swiftnet/blob/master/models/resnet/resnet_single_scale.py
