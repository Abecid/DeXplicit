"""CPU checks for projection, visibility, and the predicted-motion bridge."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from safetensors.numpy import save_file
from dexplicit.camera import Camera
from dexplicit.controls import render


class ControlsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.camera = self.root/'camera.npz'
        self.K = np.array([[80,0,50],[0,80,40],[0,0,1]],np.float32)
        np.savez(self.camera, camera_model='pinhole', world2cam=np.eye(4), intrinsics=self.K, source_hw=[80,100])
        rgb = np.zeros((80,100,3),np.uint8); rgb[:] = [170,80,30]
        Image.fromarray(rgb).save(self.root/'initial.png')
        scene = np.tile(np.array([[[[0,0,1],[0,0,2]]]],np.float32),(1,35,1,1))
        self.inputs = {'scene_flows':scene.copy(), 'scene_exists':np.ones((1,35,2,1),bool),
                       'robot_flows':np.zeros((1,35,1,3),np.float32), 'robot_exists':np.zeros((1,35,1,1),bool)}
        self.forecast = scene.copy(); self.forecast[0,1:,:,0] += .1
        self.save()

    def tearDown(self): self.tmp.cleanup()

    def save(self):
        save_file(self.inputs, str(self.root/'input.safetensors'))
        np.savez(self.root/'forecast.npz',scene_flows=self.forecast)

    def render(self):
        return render(self.root/'forecast.npz',self.root/'input.safetensors',self.root/'initial.png',self.camera)

    def test_projection_and_depth(self):
        uv,z,valid=Camera(self.camera).project(np.array([[0,0,1],[.1,0,1],[0,0,-1]]),0)
        np.testing.assert_allclose(uv[:2],[[50,40],[58,40]])
        self.assertEqual(valid.tolist(),[True,True,False])

    def test_forecast_drives_video_controls(self):
        original = self.render()[0]
        self.forecast[0,1:,:,0] += .2; self.save()
        changed = self.render()[0]
        np.testing.assert_array_equal(original[0],changed[0])
        self.assertFalse(np.array_equal(original[10],changed[10]))

    def test_future_scene_and_visibility_are_ignored(self):
        before = self.render()[0]
        self.inputs['scene_flows'][:,1:] = 1e6
        self.inputs['scene_exists'][:,1:] = False; self.save()
        np.testing.assert_array_equal(before,self.render()[0])

    def test_joint_reference_zbuffer(self):
        _,_,_,receipt=self.render()
        self.assertEqual(receipt['source_visible_points']['scene_flows'],1)
        self.assertTrue(all(x>0 for x in receipt['coverage_pixels']))

    def test_model_grid_does_not_apply_crop_twice(self):
        np.savez(self.camera, camera_model='pinhole', world2cam=np.eye(4), intrinsics=self.K,
                 source_hw=[160,200],model_hw=[80,100],pixel_transform=np.diag([.5,.5,1]))
        uv,_,_=Camera(self.camera).project(np.array([[.1,0,1]]),0)
        np.testing.assert_allclose(uv,[[58,40]])
        self.render()


    def test_legacy_intrinsics_already_transformed_without_model_hw(self):
        np.savez(self.camera, camera_model='pinhole', world2cam=np.eye(4), intrinsics=self.K,
                 source_hw=[80,100], pixel_transform=np.diag([.5,.5,1]))
        uv,_,_=Camera(self.camera).project(np.array([[.1,0,1]]),0)
        np.testing.assert_allclose(uv,[[58,40]])

    def test_reject_mismatched_initial_state(self):
        self.forecast[0,0,0,0]+=.01; self.save()
        with self.assertRaisesRegex(ValueError,'do not match'): self.render()

    def test_fisheye_optical_axis(self):
        params=np.zeros(15);params[:3]=[100,50,40]
        np.savez(self.camera,camera_model='FISHEYE624',world2cam=np.eye(4),projection_params=params,source_hw=[80,100])
        uv,_,_=Camera(self.camera).project(np.array([[0,0,1]]),0)
        np.testing.assert_allclose(uv,[[50,40]])

if __name__=='__main__':unittest.main()
