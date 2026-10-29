# ESANet Depth-Only Rice Segmentation Baseline

A single-encoder ESANet adaptation for binary rice segmentation using only three-channel pseudocolor depth images. RGB images are not required. Training starts from random initialization.

## Usage

Set data paths, device, and training parameters in `config.json`, keeping `model` as `esanet_depth`, then run:

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

Training and evaluation require depth images and binary masks. The scripts also run directly in PyCharm. Before evaluation, set `checkpoint` to the trained `best.pth`. Results are saved under `runs/`. Use new output names for repeated runs.

## Third-Party Sources and Licenses

- `context_modules.py`, `model.py`, `model_utils.py`, `resnet.py`, and `rgb_depth_fusion.py` under `official_esanet/src/models/` come from [ESANet], authored by Mona Koehler, mona.koehler@tu-ilmenau.de, and Daniel Seichter, daniel.seichter@tu-ilmenau.de. The Apache 2.0 license is provided in `official_esanet/LICENSE`.
- Parts of `context_modules.py` are adapted from [PSPNet]. Parts of `resnet.py` are adapted from [torchvision ResNet] and [SwiftNet].
- The bundled `dformerv2_encoder.py` and DFormerv2/Ham components are unused by the depth-only configuration. They are based on [DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation][DFormer], authored by yinbow, bowenyin@mail.nankai.edu.cn. The MIT license is provided in `DFORMER_LICENSE`. The local DFormerv2 adapter learns a scalar guide from pseudocolor images, not calibrated metric depth.

The code includes local adaptations. Retain the original copyright, license, and modification notices.

[ESANet]: https://github.com/TUI-NICR/ESANet
[PSPNet]: https://github.com/hszhao/semseg/blob/master/model/pspnet.py
[torchvision ResNet]: https://github.com/pytorch/vision/blob/v0.4.2/torchvision/models/resnet.py
[SwiftNet]: https://github.com/orsic/swiftnet/blob/master/models/resnet/resnet_single_scale.py
[DFormer]: https://github.com/VCIP-RGBD/DFormer
