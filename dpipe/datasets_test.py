import unittest
from dpipe import make_dataset
DATAPATH_IMAGES = 'images_dataset'
DATAPATH_VIDEOS = 'videos_dataset'
class TestFromListFactory(unittest.TestCase):
    def test_images_label(self):
        dataset = make_dataset('image', 'label', x_path=DATAPATH_IMAGES, x_size=(128,128)).build()
        self.assertEqual(len(list(dataset.as_numpy_iterator())), 6)
    def test_videos_label(self):
        dataset = make_dataset('video', 'label', x_path=DATAPATH_VIDEOS, x_size=(128,128)).build()
        self.assertEqual(len(list(dataset.as_numpy_iterator())), 4)
    # TODO: implement test cases video video, video image, image video, label video and label image
if __name__ == '__main__':
    unittest.main()
