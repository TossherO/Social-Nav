"""Offline export of the upstream OSNet x0.5 MSMT17 checkpoint for local T2."""

import argparse
import hashlib
import importlib.util
from pathlib import Path


CHECKPOINT_URL = (
    'https://huggingface.co/kaiyangzhou/osnet/resolve/main/'
    'osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth'
)
SOURCE_URL = (
    'https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/'
    'c2d9b0e8251a7eb1f8417e703a1a1d90f7fca319/torchreid/models/osnet.py'
)
CHECKPOINT_SHA256 = 'e96cbd20ee9cc3c6dcc0e8f4fbba8c8069d47647a5a96a59ce381fb785c54f68'
SOURCE_SHA256 = 'c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae'


def export(source, checkpoint, output):
    import io

    import cv2
    import numpy as np
    import torch

    source_bytes = source.read_bytes()
    checkpoint_bytes = checkpoint.read_bytes()
    for name, content, expected in ((source, source_bytes, SOURCE_SHA256),
                                    (checkpoint, checkpoint_bytes, CHECKPOINT_SHA256)):
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'Unexpected upstream SHA256 for {name}')

    spec = importlib.util.spec_from_loader('osnet_upstream', loader=None)
    upstream = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), 'exec'), upstream.__dict__)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = upstream.osnet_x0_5(num_classes=4101, pretrained=False).eval()
    model.load_state_dict(torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True))

    class Embeddings(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, images):
            features = self.network.featuremaps(images)
            pooled = self.network.global_avgpool(features).flatten(1)
            return self.network.fc(pooled)

    wrapper = Embeddings(model).eval()
    sample = torch.randn(2, 3, 256, 128)
    with torch.inference_mode():
        onnx = io.BytesIO()
        torch.onnx.export(wrapper, sample[:1], onnx, input_names=['rgb'], output_names=['embedding'],
                          dynamic_axes={'rgb': {0: 'batch'}, 'embedding': {0: 'batch'}},
                          opset_version=11, dynamo=False)
        model_bytes = onnx.getvalue()
        net = cv2.dnn.readNetFromONNX(model_bytes)
        for batch in (sample[:1], sample):
            net.setInput(batch.numpy())
            np.testing.assert_allclose(net.forward(), model(batch).numpy(), rtol=1e-3, atol=5e-4)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(model_bytes)
    print(hashlib.sha256(model_bytes).hexdigest(), output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=f'Upstream assets: {SOURCE_URL} {CHECKPOINT_URL}')
    parser.add_argument('source', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.source, args.checkpoint, args.output)


if __name__ == '__main__':
    main()
