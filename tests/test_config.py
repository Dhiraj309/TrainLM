from trainlm.config.loader import load_config

def test_load_config():
    cfg = load_config(
        "configs/smoke.yaml"
    )

    assert cfg.model.hidden_size == 512
    assert cfg.model.num_hidden_layers == 4
    assert cfg.training.max_steps == 100

def test_head_dim():
    cfg = load_config(
        "configs/smoke.yaml"
    )

    assert cfg.model.head_dim == 64

def test_gqa_ratio():
    cfg = load_config(
        "configs/smoke.yaml"
    )

    assert (
        cfg.model.num_attention_heads == 8
    )

    assert (
        cfg.model.num == 4
    )