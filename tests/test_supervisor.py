import numpy as np

from perception import Detection
from supervisor import Failure, RecoveryPolicy, SelectionPolicy, VerificationPolicy


def detection(confidence=0.9, margin=0.5, entropy=0.2, x=0.5):
    return Detection(
        0,
        "circle",
        confidence,
        margin,
        entropy,
        (320.0, 240.0),
        np.array([x, 0.0, 0.04]),
        0.0,
        0.04,
        500,
        0.001,
        False,
        np.zeros((8, 8), np.uint8),
        np.zeros((3, 1, 2), np.int32),
    )


def test_uncertain_detection_is_rejected():
    policy = SelectionPolicy()
    assert policy.select([detection(confidence=0.55, margin=0.03, entropy=0.91)]) is None
    assert policy.select([detection()]) is not None


def test_verifier_requires_two_independent_signals():
    verifier = VerificationPolicy()
    assert verifier.success(0.012, 0.04, 0.12)
    assert not verifier.success(0.002, 0.04, 0.12)
    assert not verifier.success(0.012, 0.04, 0.06)


def test_recovery_budget_and_failure_specific_action():
    policy = RecoveryPolicy(max_attempts_per_object=3)
    plan = policy.plan(Failure.EMPTY_GRASP, attempt=1)
    assert plan.retry and plan.yaw_offset_rad > 1.0 and plan.force_scale > 1.0
    assert not policy.plan(Failure.EMPTY_GRASP, attempt=3).retry
