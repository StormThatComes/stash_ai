"""SAM 3 visual prompts through Transformers; no video continuity assumptions."""
import hashlib
from .mask_backends import FrameSegmenter

REVISION = '3c879f39826c281e95690f02c7821c4de09afae7'
# Small files use their upstream git-blob IDs; safetensors use the LFS SHA256.
FILES = {
    'config.json': (25843, '3e600e765b0f1b4c52680aed9ad242c9a90f7ed1'),
    'processor_config.json': (1712, '9d3bd6df8a54852cd56087a3ae019cce051de2b2'),
    'model.safetensors': (3439938512, '6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a'),
}


def assets(config):
    from .mask_backends import asset_directory
    from .sapiensid_assets import _require_download
    root = asset_directory(config, 'sam3', REVISION)
    for name, (size, checksum) in FILES.items():
        path = root / name
        if not path.exists():
            _require_download(config, path)
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import GatedRepoError
            try:
                hf_hub_download('facebook/sam3', name, revision=REVISION, local_dir=root)
            except GatedRepoError as error:
                raise RuntimeError('SAM 3 requires approved access at https://huggingface.co/facebook/sam3 '
                                   'and an existing Hugging Face login or HF_TOKEN. After authorization '
                                   'weights download automatically; do not put tokens in recognition YAML.') from error
        digest = hashlib.sha256() if len(checksum) == 64 else hashlib.sha1()
        if len(checksum) == 40:
            digest.update(f'blob {size}\0'.encode())
        count = 0
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1048576), b''):
                count += len(chunk)
                if count > size:
                    raise ValueError('SAM 3 asset size mismatch: ' + name)
                digest.update(chunk)
        if count != size or digest.hexdigest() != checksum:
            raise ValueError('SAM 3 asset integrity mismatch: ' + name)
    return root


def load(config, device):
    try:
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    except ImportError as error:
        raise RuntimeError('SAM 3 requires transformers>=5.19,<6; install the CPU/GPU requirements.') from error
    root = assets(config)
    model, info = Sam3TrackerModel.from_pretrained(root, local_files_only=True, use_safetensors=True,
                                                   trust_remote_code=False, output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('SAM 3 checkpoint did not fully initialize the visual segmentation model.')
    model = model.float().to(device).eval()
    processor = Sam3TrackerProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    return SAM3Masks(model, processor, device, config.data['execution']['sam_precision'])


class SAM3Masks(FrameSegmenter):
    def __init__(self, model, processor, device, precision):
        super().__init__(model, device, precision)
        self.processor = processor
        self.embeddings = self.original_sizes = None

    def clear(self):
        super().clear()
        self.embeddings = self.original_sizes = None

    def encode(self, image):
        replay = getattr(image, '_tuner_mask_replay', None)
        if replay is not None and replay.encoded is not None and replay.encoded[0] is self:
            _, self.original_sizes, self.embeddings = replay.encoded
            return
        with image.convert('RGB') as rgb:
            inputs = self.processor(images=rgb, return_tensors='pt').to(self.device)
        self.original_sizes = inputs['original_sizes']
        self.embeddings = self.model.get_image_embeddings(inputs['pixel_values'])
        if replay is not None:
            replay.encoded = self, self.original_sizes, self.embeddings

    def options(self, positive, negative, multi):
        import numpy as np
        if self.embeddings is None:
            raise RuntimeError('SAM 3 requires an active image context.')
        points = np.concatenate((np.asarray(positive).reshape(-1, 2), np.asarray(negative).reshape(-1, 2)))
        labels = [1] * len(positive) + [0] * len(negative)
        inputs = self.processor(original_sizes=self.original_sizes, input_points=[[points.tolist()]],
                                input_labels=[[labels]], return_tensors='pt').to(self.device)
        outputs = self.model(image_embeddings=self.embeddings, input_points=inputs['input_points'],
                             input_labels=inputs['input_labels'], multimask_output=multi)
        masks = self.processor.post_process_masks(outputs.pred_masks.float().cpu(), self.original_sizes.cpu())[0][0]
        return self.checked(masks.numpy(), outputs.iou_scores[0, 0].float().cpu().numpy())
