FOVS = (96, 192)
POOL_SIGMA = 1.5
FEATURE_REVISION = "uni2-pixel-center-pool-v2"
MODEL_REVISION = "e16-g3-validated-v2"
SEGFORMER_BASE = dict(
    model_type="segformer",
    num_channels=3,
    num_encoder_blocks=4,
    depths=[3, 4, 6, 3],
    sr_ratios=[8, 4, 2, 1],
    hidden_sizes=[64, 128, 320, 512],
    patch_sizes=[7, 3, 3, 3],
    strides=[4, 2, 2, 2],
    mlp_ratios=[4, 4, 4, 4],
    num_attention_heads=[1, 2, 5, 8],
    hidden_act="gelu",
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    classifier_dropout_prob=0.1,
    initializer_range=0.02,
    drop_path_rate=0.1,
    layer_norm_eps=1e-06,
    decoder_hidden_size=768,
    semantic_loss_ignore_index=255,
    reshape_last_stage=True,
)


def segformer_config(n_classes):
    labels = {str(i): f"LABEL_{i}" for i in range(n_classes)}
    return {
        **SEGFORMER_BASE,
        "id2label": labels,
        "label2id": {name: int(i) for i, name in labels.items()},
    }
