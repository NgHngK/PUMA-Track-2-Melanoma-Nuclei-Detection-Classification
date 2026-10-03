class E9Error(RuntimeError):
    pass


class ManifestError(E9Error):
    pass


class ClassMapError(E9Error):
    pass


class ProtectedRoleError(E9Error):
    pass


class CoordinateError(E9Error):
    pass


class ImageContractError(E9Error):
    pass


class CheckpointIdentityError(E9Error):
    pass


class TokenGeometryError(E9Error):
    pass


class FeatureContractError(E9Error):
    pass


class CacheAlignmentError(E9Error):
    pass


class FoldLeakageError(E9Error):
    pass


class NormalizationLeakageError(E9Error):
    pass


class ParameterCountError(E9Error):
    pass


class GradientConnectivityError(E9Error):
    pass


class OOFCompletenessError(E9Error):
    pass


class MatcherParityError(E9Error):
    pass


class MetricContractError(E9Error):
    pass


class ResumeIntegrityError(E9Error):
    pass


class NonFiniteTrainingError(E9Error):
    pass


class ResourceError(E9Error):
    pass


class ScientificContractError(RuntimeError):
    pass
