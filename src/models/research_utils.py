"""Explicit loading of upstream research code; no automatic pip/git at prediction time."""
import sys
from pathlib import Path


def add_source_root(path, required):
    root = Path(path).resolve()
    if not (root / required).is_file():
        raise FileNotFoundError(f'Missing upstream code: {root / required}. See Docs/Extra_Models.md.')
    # Upstream projects use the same top-level package names. Use separate CLI
    # processes for those models, as the Kaggle pipeline already does.
    for name in ('util', 'models'):
        module = sys.modules.get(name)
        filename = getattr(module, '__file__', None)
        if filename and not Path(filename).resolve().is_relative_to(root):
            raise RuntimeError(f'Upstream package conflict: {name}. Run CountGD and VA-Count in separate processes.')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def checkpoint_path(settings):
    if settings.get('weights'):
        path = Path(settings['weights'])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(settings['hub_id'], settings['filename'],
                               revision=settings.get('revision', 'main')))


def load_state(path):
    if Path(path).suffix == '.safetensors':
        from safetensors.torch import load_file
        return load_file(str(path), device='cpu')
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    return checkpoint.get('model', checkpoint)


def load_into(network, path):
    if Path(path).suffix == '.safetensors':
        # HF safetensors may omit duplicate aliases of shared decoder weights.
        from safetensors.torch import load_model
        load_model(network, str(path), strict=True, device='cpu')
    else:
        network.load_state_dict(load_state(path), strict=True)
