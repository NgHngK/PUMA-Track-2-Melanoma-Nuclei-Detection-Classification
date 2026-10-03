from .anchor import Anchor
from .anchor_context_c import AnchorContextC
from CODE.COMMON.constants import RECONSTRUCTED_CONTEXT_SCALE_INIT


def make_model(name: str, *, context_scale_init: float = RECONSTRUCTED_CONTEXT_SCALE_INIT):
    if name == "ANCHOR":
        return Anchor()
    if name == "ANCHOR_CONTEXT_C":
        return AnchorContextC(context_scale_init=context_scale_init)
    raise KeyError(name)
