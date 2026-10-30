import numpy as np

from perception import CameraIntrinsics, PinholeCamera, look_at_transform, robust_table_depth


def test_project_backproject_round_trip():
    k = CameraIntrinsics.from_vertical_fov(640, 480, 50.0)
    transform = look_at_transform((0.55, 0.0, 1.3), (0.55, 0.0, 0.0), (1.0, 0.0, 0.0))
    camera = PinholeCamera(k, transform)
    point = np.array([0.52, 0.08, 0.04])
    uv, depth = camera.project(point)
    reconstructed = camera.backproject(uv[0], uv[1], depth)[:3]
    assert np.allclose(reconstructed, point, atol=1e-8)


def test_robust_table_depth_ignores_objects():
    depth = np.full((200, 300), 1.3, np.float32)
    depth[60:140, 100:200] = 1.255
    assert abs(robust_table_depth(depth) - 1.3) < 1e-4
