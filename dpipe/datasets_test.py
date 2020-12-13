import unittest
from dpipe import make_dataset
import tensorflow as tf
DATAPATH_IMAGES = 'images_dataset'
DATAPATH_VIDEOS = 'videos_dataset'

def make_model():
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

class TestFromListFactory(unittest.TestCase):
    def test_images_label(self):
        dataset = make_dataset('image', 'label', x_path=DATAPATH_IMAGES, x_size=(128, 128)).build()
        self.assertEqual(len(list(dataset.as_numpy_iterator())), 6)
    def test_videos_label(self):
        dataset = make_dataset('video', 'label', x_path=DATAPATH_VIDEOS, x_size=(128, 128)).build()
        self.assertEqual(len(list(dataset.as_numpy_iterator())), 4)

    def test_videos_label_cropping_single(self):
        ## TEST single
        # test external defined video frames
        dataset = make_dataset('video', 'label', x_path=DATAPATH_VIDEOS, x_size=(128, 128), video_frames=10,
                               video_cropping='single').build()
        self.assertEqual(len(list(dataset.as_numpy_iterator())), 4)
        self.assertEqual([10,128,128,3],dataset.element_spec[0].shape.as_list())
        # test inferred out video frames
        dataset = make_dataset('video', 'label', x_path=DATAPATH_VIDEOS, x_size=(128, 128),
                               video_cropping='single').build()
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
        dataset = make_dataset('video', 'label', x_path=DATAPATH_VIDEOS, x_size=(128, 128),
                               video_cropping='multi').build()
        self.assertEqual(len(list(dataset.as_numpy_iterator())), 4)
        self.assertEqual([140, 128, 128, 3], dataset.element_spec[0].shape.as_list())

    def test_parallel_training(self):
        EPOCHS = 10
        LENGTH = 50
        dataset_builder = make_dataset('image', 'label', x_path=DATAPATH_IMAGES, x_size=(128,128), one_hot_encoding=False)

        dataset = dataset_builder.\
            shuffle(LENGTH, reshuffle_each_iteration=True). \
            batch(1).\
            repeat(EPOCHS).\
            build()


        print(dataset.element_spec)

        model = make_model()
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.RMSprop())
        print("Build arguments: ", dataset.built_args)
        model.fit(x=dataset, epochs=EPOCHS, **dataset.built_args)

    # TODO: implement test cases video video, video image, image video, label video and label image
if __name__ == '__main__':
    unittest.main()
