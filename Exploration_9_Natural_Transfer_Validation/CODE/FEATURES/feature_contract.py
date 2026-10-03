from __future__ import annotations
from dataclasses import asdict
from CODE.COMMON.constants import CLASSES, APPEARANCE_DIM, GAUSSIAN_SIGMA
from CODE.COMMON.hashing import canonical_json_hash
from CODE.COMMON.schemas import FeatureContract


def class_map_hash():
    return canonical_json_hash(list(CLASSES))


def build_contract(feature_name: str, fov: int, encoder_sha256: str, input_contract_hash: str):
    return FeatureContract(
        feature_name,
        int(fov),
        float(GAUSSIAN_SIGMA),
        APPEARANCE_DIM,
        encoder_sha256,
        input_contract_hash,
        class_map_hash(),
    )


def contract_hash(contract: FeatureContract):
    return canonical_json_hash(asdict(contract))
