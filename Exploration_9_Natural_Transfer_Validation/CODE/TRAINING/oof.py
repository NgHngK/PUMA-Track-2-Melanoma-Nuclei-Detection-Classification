from __future__ import annotations
from CODE.COMMON.exceptions import OOFCompletenessError


def validate_oof(uids, records, seed, arm):
    expected = set(uids)
    got = [r['uid'] for r in records if r['seed'] == seed and r['arm'] == arm]
    if len(got) != len(set(got)):
        raise OOFCompletenessError(f'Duplicate OOF UID for {seed}/{arm}')
    if set(got) != expected:
        raise OOFCompletenessError(
            f'OOF mismatch missing={len(expected-set(got))} extra={len(set(got)-expected)}'
        )
    return True
