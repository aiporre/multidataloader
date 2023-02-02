from unittest import TestCase
import os
import numpy as np
from dpipe.video_reader import _cv2_vread

VIDEO_PATH = "./videos_dataset/tennis_swing/v_tennis_08_06.avi"
VIDEO_PATH_2 = "./dpipe/videos_dataset/tennis_swing/v_tennis_08_06.avi"
class Test(TestCase):

    def setUp(self) -> None:
        self.video_path = VIDEO_PATH if os.path.exists(VIDEO_PATH) else VIDEO_PATH_2
    def test__cv2_vread(self):
        video = _cv2_vread(self.video_path)
        self.assertTupleEqual(video.shape, (151, 240, 320, 3))
        self.assertEqual(video.dtype, np.dtype('uint8'))

