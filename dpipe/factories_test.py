import unittest
from dpipe import *
import numpy as np
from random import choice
from string import ascii_letters, digits
import tensorflow as tf


class Counter(object):
    """ Dummy counter class
    """
    def __init__(self):
        self.values = [1,2,3,4]
    def __len__(self):
        return 4
    def read(self,index):
        print('index at counter', index)
        return self.values[index]

class ListNumerics(object):
    """ Dummy counter class
    """
    def __init__(self):
        self.values = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    def __len__(self):
        return 3
    def read(self,index):
        print('index at counter', index)
        return tuple(self.values[index])

def make_random_string(lenght=None):
    if lenght is None:
        lenght = np.random.randint(15)
    return ''.join([choice(ascii_letters+digits) for _ in range(lenght)])
class Strings(object):
    """ Dummy counter class
    """
    def __init__(self):
        self.values = [make_random_string() for _ in range(10)]
    def __len__(self):
        return 10
    def read(self,index):
        print('index at counter', index)
        return self.values[index]
class ListOfStrings(object):
    """ Dummy counter class
    """
    def __init__(self):
        self.values = [[make_random_string() for _ in range(3)] for _ in range(10)]
    def __len__(self):
        return 10
    def read(self,index):
        print('index at counter', index)
        return tuple(self.values[index])
class ImageWithLabel(object):
    """ Dummy counter class
    """
    def __init__(self):
        self.values = 10*[(np.random.rand(3,10,10),np.random.randint(9))]
    def __len__(self):
        return 10
    def read(self,index):
        print('index at counter', index)
        return self.values[index]
class DoubleImage(object):
    """ Dummy counter class
    """
    def __init__(self,length=10):
        self.values = length*[(np.random.rand(3,10,10),np.random.rand(3,10,10))]
        self.length = length
    def __len__(self):
        return self.length
    def read(self,index):
        print('index at counter', index)
        return self.values[index]
class DoubleImageList(object):
    """ Dummy counter class
    """
    def __init__(self):
        self.values = 10*[[np.random.rand(3,10,10),np.random.rand(3,10,10)]]
    def __len__(self):
        return 10
    def read(self,index):
        print('index at counter', index)
        return self.values[index]
def get_all_methods(object):
    return [method_name for method_name in dir(object) if callable(getattr(object, method_name))]

class TestFactoriesMethods(unittest.TestCase):

    def test_from_object_one_output(self):
        c = Counter()
        dataset_builder = from_object(c,'read')
        dataset = dataset_builder.build()
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        self.assertEqual(0, 0)

    def test_from_object_two_outputs(self):
        imgs = ImageWithLabel()
        dataset_builder = from_object(imgs,'read')
        dataset = dataset_builder.build()
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        imgs = DoubleImage()
        dataset_builder = from_object(imgs,'read')
        dataset = dataset_builder.build()
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
    def test_from_object_two_outputs_list(self):

        imgs = DoubleImageList()
        dataset_builder = from_object(imgs,'read')
        dataset = dataset_builder.build()
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        self.assertEqual(0, 0)
    def test_from_object_string_outputs(self):
        strs = Strings()
        dataset_builder = from_object(strs,'read')
        dataset = dataset_builder.build()
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        self.assertEqual(0, 0)
    def test_from_object_list_string_outputs(self):
        strs = ListOfStrings()
        dataset_builder = from_object(strs,'read')
        dataset = dataset_builder.build()
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        self.assertEqual(0, 0)
    def test_from_object_list_numerics_outputs(self):
        numerics = ListNumerics()
        dataset_builder = from_object(numerics,'read')
        dataset = dataset_builder.build()
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        self.assertEqual(0, 0)


class ImageFlatWithLabel(object):
    """ Dummy counter class
    """
    def __init__(self,length=10):
        # image 10x10 and a label
        self.values = length*[(np.random.rand(100),np.random.randint(9))]
        self.length = length
    def __len__(self):
        return self.length
    def read(self,index):
        print('index at counter', index)
        return self.values[index]

def make_model():
    inputs = tf.keras.Input(shape=(100,))
    dense = tf.keras.layers.Dense(64, activation='relu')
    x = dense(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


class TestFactoryDatasetBuilder(unittest.TestCase):
    def test_build(self):
        c = DoubleImage(length=100)
        dataset_builder = from_object(c,'read')
        dataset = dataset_builder.batch(20).cache().build()
        print("Build arguments: ", dataset.built_args)
        print("element spec: ", dataset.element_spec )
        # for element in dataset:
        #     print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        self.assertEqual(0, 0)
    def test_build(self):
        LENGTH = 50
        c = ImageFlatWithLabel(length=LENGTH)
        dataset_builder = from_object(c,'read')
        dataset = dataset_builder.build()
        self.assertEqual(len(list(dataset.as_numpy_iterator())) ,LENGTH)
    def test_batching(self):
        BATCH = 10
        LENGTH = 50
        c = ImageFlatWithLabel(length=LENGTH)
        dataset_builder = from_object(c, 'read')
        dataset = dataset_builder.batch(BATCH).build()
        self.assertEqual(len(list(dataset.as_numpy_iterator())), LENGTH//BATCH)
    def test_shuffle(self):
        c = Counter()
        dataset_builder = from_object(c,'read')
        dataset = dataset_builder.shuffle(len(c), reshuffle_each_iteration=True).build()
        list1 = list(dataset.as_numpy_iterator())
        list2 = list(dataset.as_numpy_iterator())
        self.assertFalse(list1==list2)
    def test_training(self):
        EPOCHS = 10
        LENGTH = 50
        c = ImageFlatWithLabel(length=LENGTH)
        dataset_builder = from_object(c,'read')
        dataset = dataset_builder.shuffle(LENGTH, reshuffle_each_iteration=True).batch(2).repeat().build()
        model = make_model()
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop())
        print("Build arguments: ", dataset.built_args)
        model.fit(x=dataset, epochs=EPOCHS,**dataset.built_args)

if __name__ == '__main__':
    unittest.main()
