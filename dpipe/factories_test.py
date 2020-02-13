import unittest
from factories import *
import numpy as np

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

def get_all_methods(object):
    return [method_name for method_name in dir(object) if callable(getattr(object, method_name))]

class TestFactoriesMethods(unittest.TestCase):

    def test_from_object_one_output(self):
        c = Counter()
        dataset = from_object(c,'read')
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        self.assertEqual(0, 0)

    def test_from_object_two_outputs(self):
        imgs = ImageWithLabel()
        dataset = from_object(imgs,'read')
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        self.assertEqual(0, 0)

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
