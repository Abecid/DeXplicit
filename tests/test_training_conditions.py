"""Prevent measured/future geometry from silently entering renderer training."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from safetensors.numpy import save_file
from dexplicit.conditioning import (TRAINING_FORMAT, sha256, validate_training_bundle,
                                    validate_training_row)


class TrainingConditionsTest(unittest.TestCase):
    def make_bundle(self, root, *, source='predicted', future=False, labelled=True):
        controls = root/'predicted-controls'
        controls.mkdir()
        (controls/'forecast.npz').write_bytes(b'forecast-fixture')
        (controls/'controls.npz').write_bytes(b'projected-control-fixture')
        receipt = {'schema': 'dexplicit-predicted-controls-v1', 'scene_source': source,
                   'producer': 'dexplicit.geometry.infer',
                   'future_scene_or_rgb_input': future, 'point_positions_modified': False,
                   'forecast_file': 'forecast.npz', 'controls_file': 'controls.npz',
                   'forecast_sha256': sha256(controls/'forecast.npz'),
                   'controls_sha256': sha256(controls/'controls.npz'),
                   'inputs': {'geometry_weights': {'sha256': 'a'*64}}}
        rp = controls/'controls.json'
        rp.write_text(json.dumps(receipt))
        meta = {'format': TRAINING_FORMAT, 'conditioning_source': 'predicted_3d',
                'conditioning_receipt': 'predicted-controls/controls.json',
                'conditioning_receipt_sha256': sha256(rp), 'geometry_weights_sha256': 'a'*64}
        tensors = {key: np.zeros(shape, dtype=np.float16) for key, shape in {
            'context': (1, 512, 4096), 'vace_context': (1, 96, 10, 60, 104),
            'input_latents': (1, 16, 10, 60, 104)}.items()}
        path = root/'sample.safetensors'
        save_file(tensors, str(path), metadata=meta if labelled else {})
        return path

    def test_preserved_prediction_artifacts_are_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.make_bundle(Path(d))
            self.assertEqual(validate_training_bundle(path)['conditioning_source'], 'predicted_3d')

    def test_measured_or_future_conditioning_is_rejected(self):
        for source, future in [('measured', False), ('predicted', True)]:
            with self.subTest(source=source, future=future), tempfile.TemporaryDirectory() as d:
                path = self.make_bundle(Path(d), source=source, future=future)
                with self.assertRaisesRegex(ValueError, 'unchanged DeXplicit 3D model forecast'):
                    validate_training_bundle(path)

    def test_unlabelled_legacy_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.make_bundle(Path(d), labelled=False)
            with self.assertRaisesRegex(ValueError, 'measured or unlabelled'):
                validate_training_bundle(path)

    def test_changed_forecast_or_controls_are_rejected(self):
        for name in ['forecast.npz', 'controls.npz']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                root = Path(d); path = self.make_bundle(root)
                (root/'predicted-controls'/name).write_bytes(b'replaced-after-encoding')
                with self.assertRaisesRegex(ValueError, 'artifact checksum mismatch'):
                    validate_training_bundle(path)

    def test_precomputed_controls_and_nontraining_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for key in ['geometry_input', 'initial_image', 'camera', 'video']:
                (root/key).touch()
            row = {key: key for key in ['geometry_input', 'initial_image', 'camera', 'video']}
            row['prompt'] = 'hand object interaction'
            self.assertEqual(validate_training_row(row, root)['video'], (root/'video').resolve())
            with self.assertRaisesRegex(ValueError, 'precomputed controls'):
                validate_training_row({**row, 'controls': 'measured.npz'}, root)
            with self.assertRaisesRegex(ValueError, 'training split'):
                validate_training_row({**row, 'split': 'test'}, root)


if __name__ == '__main__':
    unittest.main()
