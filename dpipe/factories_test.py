import unittest
from factories import *
import numpy as np
from random import choice
from string import ascii_letters, digits
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
    #
    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)
class TestFactoryDatasetBuilder(unittest.TestCase):
    def test_training(self):
        c = DoubleImage(length=100)
        dataset_builder = from_object(c,'read')
        dataset = dataset_builder.batch(20).cache().build()
        print("Build arguments: ", dataset.built_args)
        print("element spec: ", dataset.element_spec )
        # for element in dataset:
        #     print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        self.assertEqual(0, 0)
if __name__ == '__main__':
    unittest.main()
