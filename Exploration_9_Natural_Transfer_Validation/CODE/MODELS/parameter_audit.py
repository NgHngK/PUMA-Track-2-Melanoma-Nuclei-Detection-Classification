EXPECTED = {
    "ANCHOR": 27866,
    "ANCHOR_CONTEXT_C": 52523,
}


def assert_parameter_count(model, name: str) -> int:
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = EXPECTED[name]
    if count != expected:
        raise RuntimeError(f"{name}: expected {expected} trainable parameters, got {count}")
    return count
