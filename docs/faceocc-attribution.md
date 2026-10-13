# Face visibility model attribution

The inference-only U-Net/ResNet18 adapter in `stash_ai/recognition/faceocc.py` follows the state-dict and decoder contract of segmentation-models-pytorch v0.5.0. The application uses its existing torch/torchvision dependencies and does not download ImageNet encoder weights.

## Segmentation Models PyTorch

Source: https://github.com/qubvel-org/segmentation_models.pytorch/tree/v0.5.0

The MIT License

Copyright (c) 2019, Pavel Iakubovskii

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## FaceOcc

Pinned checkpoint: https://huggingface.co/mertakin/FaceOcc/tree/03f229dc75fa14ae480cca9810f983912c2730ad

Original method: Xiangnan Yin and Liming Chen, *FaceOcc: A Diverse, High-quality Face Occlusion Dataset for Human Face Extraction*, TAIMA 2022 / arXiv:2201.08425. Source: https://github.com/face3d0725/FaceExtraction

Modernization/checkpoint release: Mert Akın (@mertakinstd), FaceOcc v1.0.0. Source: https://github.com/mertakinstd/FaceOcc ; archived release: https://doi.org/10.5281/zenodo.22261117

The modernized software additionally identifies Copyright (c) 2026 Mert Akın. The upstream copyright/permission notice is reproduced below. The model release describes the checkpoint as MIT licensed; third-party datasets and resources retain their own terms. This application downloads the pinned inference checkpoint, not training datasets.

Copyright (c) 2024 Xiangnan Yin (ECL Liris), Di Huang (Beihang University), Liming Chen (ECL Liris)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
