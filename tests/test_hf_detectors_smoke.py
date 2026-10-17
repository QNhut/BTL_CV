"""Real Transformers forward/backward with tiny randomly initialized local models.

These tests never download the production checkpoints.
"""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers', minversion='5.20.1')

from src.models import build_model


@pytest.mark.parametrize('kind', ['rtdetrv2', 'rf_detr'])
def test_transformers_detector_local_training_and_inference(kind, tmp_path):
    if kind == 'rtdetrv2':
        from transformers import RTDetrResNetConfig, RTDetrV2Config, RTDetrV2ForObjectDetection, RTDetrImageProcessor
        backbone = RTDetrResNetConfig(embedding_size=8, hidden_sizes=[8, 16, 32, 64],
                                     depths=[1, 1, 1, 1], layer_type='basic', out_indices=[2, 3, 4])
        cfg = RTDetrV2Config(backbone_config=backbone, num_labels=1, d_model=32,
                             encoder_hidden_dim=32, encoder_in_channels=[16, 32, 64],
                             decoder_in_channels=[32, 32, 32], encoder_ffn_dim=64,
                             decoder_ffn_dim=64, decoder_layers=2, num_queries=5,
                             encoder_attention_heads=4, decoder_attention_heads=4,
                             num_denoising=0)
        network = RTDetrV2ForObjectDetection(cfg)
        processor = RTDetrImageProcessor(size={'height': 64, 'width': 64})
    else:
        from transformers import RfDetrDinov2Config, RfDetrConfig, RfDetrForObjectDetection, RfDetrImageProcessor
        backbone = RfDetrDinov2Config(image_size=64, patch_size=8, hidden_size=32,
                                     num_hidden_layers=4, num_attention_heads=4, num_windows=1,
                                     out_indices=[1, 2, 3, 4])
        cfg = RfDetrConfig(backbone_config=backbone, num_labels=1, d_model=32,
                           decoder_layers=1, decoder_ffn_dim=64, num_queries=5,
                           decoder_self_attention_heads=4, decoder_cross_attention_heads=4,
                           group_detr=1, c2f_num_blocks=1, num_feature_levels=1)
        network = RfDetrForObjectDetection(cfg)
        processor = RfDetrImageProcessor(size={'height': 64, 'width': 64})
    network.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)
    model = build_model({'type': kind, 'mode': 'finetune', 'model': {'hub_id': str(tmp_path), 'device': 'cpu'},
                         'inference': {'min_score': 0.05}})
    image = torch.rand(3, 64, 64)
    target = {'boxes': torch.tensor([[10., 10., 30., 35.]])}
    losses = model.training_step(([image, image], [target, target]))
    assert torch.isfinite(losses['loss'])
    losses['loss'].backward()
    result = model.predict_image(np.zeros((64, 64, 3), np.uint8))
    assert result.boxes.shape[1] == 4
    assert len(result.boxes) == len(result.scores)


def test_grounding_dino_local_prompted_inference(tmp_path):
    from transformers import (GroundingDinoConfig, GroundingDinoForObjectDetection,
                              GroundingDinoProcessor, GroundingDinoImageProcessor,
                              BertConfig, BertTokenizer, SwinConfig)
    text = BertConfig(vocab_size=8, hidden_size=32, num_hidden_layers=1,
                      num_attention_heads=4, intermediate_size=64)
    vision = SwinConfig(image_size=64, embed_dim=8, depths=[1, 1, 1, 1],
                        num_heads=[1, 2, 4, 8], window_size=2, out_indices=[2, 3, 4])
    cfg = GroundingDinoConfig(backbone_config=vision, text_config=text, d_model=64,
                              encoder_layers=1, decoder_layers=2, encoder_ffn_dim=64,
                              decoder_ffn_dim=64, encoder_attention_heads=4, decoder_attention_heads=4,
                              num_queries=5, max_text_len=16, disable_custom_kernels=True)
    network = GroundingDinoForObjectDetection(cfg)
    tokenizer = BertTokenizer(vocab={'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
                                     '[MASK]': 4, 'apple': 5, '.': 6, 'fruit': 7})
    processor = GroundingDinoProcessor(image_processor=GroundingDinoImageProcessor(
        size={'shortest_edge': 64, 'longest_edge': 64}), tokenizer=tokenizer)
    network.save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)
    model = build_model({'type': 'grounding_dino', 'mode': 'pretrained',
                         'model': {'hub_id': str(tmp_path), 'prompt': 'apple.', 'device': 'cpu'},
                         'inference': {'min_score': 0.05}})
    result = model.predict_image(np.zeros((64, 64, 3), np.uint8))
    assert result.boxes.shape[1] == 4
    assert np.isfinite(result.scores).all()
