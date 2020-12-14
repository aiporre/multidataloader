import unittest
from dpipe import *
import numpy as np
from random import choice
from string import ascii_letters, digits
import tensorflow as tf
import time
from dpipe.factories import from_function


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
        return len(self.values)
    def read(self,index):
        print('index at counter', index, 'type', type(index))
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

class GeneratorSingleListOfNumbers(object):
    """
    Dummy counter class with generator
    """
    def __init__(self):
        self.values = list(range(10))
    def __len__(self):
        return 10
    def read(self,index):
        # returns a counter generator over the list from 0 to the given index.
        i = 0
        index = 9 if index>10 else index
        while i <= index:
            yield self.values[i]
            i += 1



def get_all_methods(object):
    return [method_name for method_name in dir(object) if callable(getattr(object, method_name))]

class TestFactoriesMethods(unittest.TestCase):

    def test_from_object_one_output(self):
        c = Counter()
        dataset_builder = from_object(c,'read','values')
        dataset = dataset_builder.build()
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        self.assertEqual(0, 0)

    def test_from_object_two_outputs(self):
        imgs = ImageWithLabel()
        dataset_builder = from_object(imgs,'read','values')
        dataset = dataset_builder.build()
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        imgs = DoubleImage()
        dataset_builder = from_object(imgs,'read','values')
        dataset = dataset_builder.build()
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
    def test_from_object_two_outputs_list(self):

        imgs = DoubleImageList()
        dataset_builder = from_object(imgs,'read','values')
        dataset = dataset_builder.build()
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        self.assertEqual(0, 0)
    def test_from_object_string_outputs(self):
        strs = Strings()
        dataset_builder = from_object(strs,'read','values')
        dataset = dataset_builder.build()
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        self.assertEqual(0, 0)
    def test_from_object_list_string_outputs(self):
        strs = ListOfStrings()
        dataset_builder = from_object(strs,'read','values')
        dataset = dataset_builder.build()
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        self.assertEqual(0, 0)
    def test_from_object_list_numerics_outputs(self):
        numerics = ListNumerics()
        dataset_builder = from_object(numerics,'read','values')
        dataset = dataset_builder.build()
        # print(list(dataset))
        print("element spec: ", dataset.element_spec )
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        self.assertEqual(0, 0)
    def test_from_object_with_generator(self):
        genSingleList = GeneratorSingleListOfNumbers()
        dataset_builder = from_object(genSingleList, getitem_fcn='read',itemlist_name='values')
        dataset = dataset_builder.build()
        # print(list(dataset))
        print("element spec: ", dataset.element_spec)
        for element in dataset:
            print('element of the dataset: ', element)
            # print('methods: ', get_all_methods(element))
        self.assertEqual(0, 0)

class ImageFlatWithLabel(object):
    """ Dummy counter class
    """
    def __init__(self,length=10):
        # image 10x10 and a label
        self.list = length*[(np.random.rand(100),np.random.randint(9))]
        self.length = length
    def __len__(self):
        return self.length
    def read(self,index):
        print('index at counter', index)
        return self.list[index]


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
        dataset_builder = from_object(c,'read','values')
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

TEXT_FILE = 'text_dataset/text_emotion.csv'

def csv_to_lists(csv_file):
    with open(csv_file) as f:
        line = f.readline()
        # it skips header
        line = f.readline()
        emotions, messages = [], []
        while line:
            tokens = line.strip().split(',')
            emotions.append(tokens[1])
            messages.append(' '.join(tokens[3:]).replace('"',''))
            line = f.readline()
    return messages, emotions

def compute_dictionary(tokens):
    codes_dict = {c: i for i, c in enumerate(set(tokens))}
    return codes_dict

def text_encode(message, codes_dict):
    return np.array([codes_dict[word] for word in message.split(' ')], dtype=int)

class TestFactoryFunctionList(unittest.TestCase):
    def test_read_idem(self):
        # extract messages and emotions from csv file
        messages, emotions = csv_to_lists(TEXT_FILE)
        dataset_builder = from_function(lambda x: x, list(zip(messages,emotions)))
        dataset = dataset_builder.build()
        self.assertEqual(len(list(dataset.as_numpy_iterator())), 40000)
        self.assertEqual(str(dataset.element_spec),
                         '(TensorSpec(shape=(), dtype=tf.string, name=None), TensorSpec(shape=(), dtype=tf.string, '
                         'name=None))')
    def test_read_function_float_gen(self):
        def counter_float(v):
            return 1.0

        def counter_float_ndarray(v):
            return np.array(1.0)

        dataset = from_function(counter_float, [1,2,3,4]).build()
        print(list(dataset.as_numpy_iterator()))
        dataset = from_function(counter_float_ndarray, [1, 2, 3, 4]).build()
        print(list(dataset.as_numpy_iterator()))

    def test_read_text_file(self):
        def read_fcn(filename):
            messages, emotions = csv_to_lists(filename)
            # find message unique dictionary encoding
            message_dict = compute_dictionary(' '.join(messages).split(' '))
            emotions_dict = compute_dictionary(' '.join(emotions).split(' '))
            encoding_x_fnc = lambda x: text_encode(x, message_dict)
            encoding_y_fnc = lambda x: text_encode(x, emotions_dict)
            messages = list(map(encoding_x_fnc,messages))
            emotions = list(map(encoding_y_fnc, emotions))
            for m,e in zip(messages, emotions):
                yield m,e
        N = 2
        filenames = N*[TEXT_FILE]
        # comparing the tf normal  and the implementation of file concurrent reading with dpipe function way
        dataset_builder = from_function(read_fcn, filenames,
                                        undetermined_shape=([0], []) # messages are shape [1] with undetermined shape at the dim 0
                                                                     # the second correspond to the emotions which is an scalar text
                                        )
        t0 = time.time()
        dataset = dataset_builder.build()
        self.assertEqual(len(list(dataset.as_numpy_iterator())), N*40000)
        t1 = time.time()-t0
        print('dadspipe implementation took: ', t1, 'seconds')

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

    def test_parallel_training(self):
        EPOCHS = 10
        LENGTH = 50
        c = ImageFlatWithLabel(length=LENGTH)
        dataset_builder = from_object(c,'read')
        dataset = dataset_builder.shuffle(LENGTH, reshuffle_each_iteration=True).batch(2).repeat().\
            parallelize_extraction(num_parallel_calls=2).build()
        model = make_model()
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop())
        print("Build arguments: ", dataset.built_args)
        model.fit(x=dataset, epochs=EPOCHS,**dataset.built_args)

if __name__ == '__main__':
    unittest.main()
