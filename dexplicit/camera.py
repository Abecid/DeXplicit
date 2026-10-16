# SPDX-License-Identifier: Apache-2.0
"""Calibrated projection; coordinates and camera translations are in metres."""
import numpy as np

def _project_fisheye624_numpy(points_cam: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_cam, dtype=np.float32)
    params = np.asarray(params, dtype=np.float32).reshape(-1)
    z = pts[:, 2]
    valid_z = np.isfinite(z) & (z > 1e-6)
    safe_z = np.where(valid_z, z, 1.0)
    x = pts[:, 0] / safe_z
    y = pts[:, 1] / safe_z
    rho = np.sqrt(np.maximum(x * x + y * y, 0.0))
    theta = np.arctan(rho)
    inv_rho = np.zeros_like(rho, dtype=np.float32)
    np.divide(1.0, rho, out=inv_rho, where=rho > 1e-6)
    cos_phi = x * inv_rho
    sin_phi = y * inv_rho

    radial = theta.copy()
    theta_power = theta.copy()
    theta2 = theta * theta
    for coeff in params[3:9]:
        theta_power = theta_power * theta2
        radial = radial + coeff * theta_power

    u_r = radial * cos_phi
    v_r = radial * sin_phi
    r2 = radial * radial
    p0, p1 = params[9], params[10]
    s0, s1, s2, s3 = params[11], params[12], params[13], params[14]
    tan_x = p0 * (2.0 * u_r * u_r + r2) + 2.0 * p1 * u_r * v_r
    tan_y = p1 * (2.0 * v_r * v_r + r2) + 2.0 * p0 * u_r * v_r
    prism_x = s0 * r2 + s1 * r2 * r2
    prism_y = s2 * r2 + s3 * r2 * r2

    pixels = np.stack(
        [
            params[0] * (u_r + tan_x + prism_x) + params[1],
            params[0] * (v_r + tan_y + prism_y) + params[2],
        ],
        axis=-1,
    ).astype(np.float32)
    depth = np.where(valid_z, z, 0.0).astype(np.float32)
    return pixels, depth, valid_z

class Camera:
    """Projection calibration is expressed on the supplied initial-image grid.

    Legacy pixel_transform is provenance: intrinsics already describe the grid. The same
    clockwise rotation and aspect-preserving center crop are then applied to
    both the image and raster, never to the metric trajectories.
    """
    def __init__(self, path, frames=33):
        with np.load(path, allow_pickle=False) as z:
            self.data = {k: z[k] for k in z.files}
        d = self.data
        self.model = str(d['camera_model'].item()).lower()
        if self.model not in {'pinhole', 'fisheye624'}:
            raise ValueError('camera_model must be pinhole or FISHEYE624')
        self.frames = frames
        self.poses = self.sequence('world2cam', (4, 4))
        if not np.allclose(self.poses[:, 3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError('world2cam must contain homogeneous rigid transforms')
        r = self.poses[:, :3, :3]
        if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=5e-3) or not np.allclose(np.linalg.det(r), 1, atol=5e-3):
            raise ValueError('Invalid camera rotation')
        # Existing processed pinhole sidecars store K in model pixels already.
        model_grid = self.model == 'pinhole' and 'model_hw' in d
        self.hw = self.sequence('model_hw' if model_grid else 'source_hw', (2,)).astype(int)
        if (self.hw < 1).any() or not np.all(self.hw == self.hw[0]):
            raise ValueError('source_hw must be a fixed positive image size')
        self.calibration = self.sequence('intrinsics', (3, 3)) if self.model == 'pinhole' else self.sequence('projection_params', (15,))
        self.rotate = bool(np.asarray(d.get('rotate_clockwise_90', False)).item())
        if model_grid and self.rotate: raise ValueError('Model-grid pinhole images must already be oriented')

    def sequence(self, key, tail):
        x = np.asarray(self.data[key], dtype=np.float32)
        if x.shape == tail: x = np.repeat(x[None], self.frames, axis=0)
        if x.shape != (self.frames, *tail) or not np.isfinite(x).all():
            raise ValueError(f'{key} must be {tail} or {(self.frames, *tail)}, finite')
        return x

    def project(self, points, t):
        p = np.asarray(points, dtype=np.float32)
        cp = np.concatenate((p, np.ones((len(p), 1), dtype=np.float32)), 1) @ self.poses[t].T
        if self.model == 'fisheye624':
            uv, depth, valid = _project_fisheye624_numpy(cp[:, :3], self.calibration[t])
        else:
            depth = cp[:, 2]
            valid = np.isfinite(cp).all(1) & (depth > 1e-6)
            h = cp[:, :3] @ self.calibration[t].T
            uv = h[:, :2] / np.where(valid, h[:, 2], 1)[:, None]
        return uv.astype(np.float32), depth, valid & np.isfinite(uv).all(1)
