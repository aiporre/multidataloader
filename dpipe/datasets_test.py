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

    def test_videos_label_cropping_single(self):
        ## TEST single
        # test external defined video frames
        dataset = make_dataset('video', 'label', x_path=DATAPATH_VIDEOS, x_size=(128, 128), video_frames=10,
                               video_cropping='single').build()
        self.assertEqual(len(list(dataset.as_numpy_iterator())), 4)
        self.assertEqual([10,128,128,3],dataset.element_spec[0].shape.as_list())
        # test inferred out video frames
        dataset = make_dataset('video', 'label', x_path=DATAPATH_VIDEOS, x_size=(128, 128), video_cropping='single').build()
        self.assertEqual(len(list(dataset.as_numpy_iterator())), 4)
        self.assertEqual([140, 128, 128, 3], dataset.element_spec[0].shape.as_list())

    def test_videos_label_cropping_multi(self):
        ## TEST multi
        # test external defined video frames
        dataset = make_dataset('video', 'label', x_path=DATAPATH_VIDEOS, x_size=(128, 128), video_frames=10,
                               video_cropping='multi').recompute_length().build()
        for m,n in dataset.as_numpy_iterator():
            self.assertEqual((10, 128, 128, 3), m.shape)
            self.assertEqual((), n.shape)
            break
        self.assertEqual(58, len(list(dataset.as_numpy_iterator())))
        self.assertEqual(58, dataset.length)

        self.assertEqual([10,128,128,3],dataset.element_spec[0].shape.as_list())
        # test inferred out video frames
        dataset = make_dataset('video', 'label', x_path=DATAPATH_VIDEOS, x_size=(128, 128), video_cropping='multi').build()
        self.assertEqual(len(list(dataset.as_numpy_iterator())), 4)
        self.assertEqual([140, 128, 128, 3], dataset.element_spec[0].shape.as_list())

    # TODO: implement test cases video video, video image, image video, label video and label image
if __name__ == '__main__':
    unittest.main()
