from warnings import warn

try:
    import tensorflow as tf
except Exception as e:
    print('tensorflow import failed', e)
try:
    from utils import get_tf_dtype, get_tf_shape, is_iterable
except:
    from .utils import get_tf_dtype, get_tf_shape, is_iterable
import numpy as np
import types


def _get_types_output(element_spec):
    shapes = tuple(e.shape.as_list() for e in element_spec)
    types = tuple(e.dtype for e in element_spec)
    return types, shapes

class _Reader(object):
    def __init__(self, read_fcn, list):
        self.length = len(list)
        self.read_fcn = read_fcn
        self.list = list
    def __len__(self):
        return self.length
    def __getitem__(self, item):
        return self.read_fcn(self.list[item])



class AugmentedDataset(object):
    """ Augments :class:`tf.data.Dataset` to handle custom configurations

    :param dataset: Instance of a :class:`tf.data.Dataset`
    :type dataset: :class:`tf.data.Dataset`
    :param length: length of the dataset, defaults to None
    :type length: int, optional
    :param training: defines training/validation flag. If `True` then the augmented dataset handles training configurations, and if `False` the augmented dataset handles validation configurations, defaults to `True`
    :type training: bool, optional
    """
    def __init__(self,dataset, gen_object=None, length=None,training=True):
        self.dataset = dataset
        self.batch_size = 1
        self.length = length
        self.training = training
        self.gen_object = gen_object

    def _build_argments_fit(self):
        ''' Creates the arguments used in fit
        '''
        if self.training:
            arguments = {'batch_size':None,
                    'steps_per_epoch':self.length}
        else:
            arguments = {'validation_steps':self.length}
        self.dataset.built_args = arguments
        self.dataset.length = self.length

    def recompute_length(self):
        """
        Recompute the length of the datatase.

        This may take long since all the samples must be accessed.
        """
        warn('Recomputing the length may take a long time depending on the data access.')
        self.length = 0
        for _ in self.dataset:
            self.length += 1
        return self

    def batch(self,batch_size):
        '''Make dataset batchs of specific batch size

        :param batch_size: size of the batches
        :type batch_size: int
        '''
        self.batch_size = batch_size
        self.dataset = self.dataset.batch(batch_size)
        self.length //= self.batch_size
        return self
    def cache(self,filename=''):
        '''Defines the cache file to store previously loaded samples

        :param filename: File name of the file where the loaded samples are stored. The second access will be loaded from the cache, defaults to ''
        :type filename: str, optional
        '''
        self.dataset = self.dataset.cache(filename=filename)
        return self
    def enumerate(self,start=0):
        ''' As the build-in enumerate function creates an index next to the sample.

        :param start: start count of the enumeration, defaults to 0
        :type start: int, optional
        '''
        self.dataset = self.dataset.enumerate(start=start)
        return self
    def filter(self,filter_fcn):
        ''' Applies a filter function to all the samples of dataset. Applies lazily.

        :param filter_fcn: funcion reference
        '''
        self.dataset = self.dataset.filter(filter_fcn)
        return self
    def map(self,map_func,num_parallel_calls=None):
        '''Maps every sample in the dataset by a map function.

        :param map_func: function reference
        '''
        if num_parallel_calls is None:
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        self.dataset = self.dataset.map(
                                    map_func,
                                    num_parallel_calls=num_parallel_calls)
        return self
    def parallelize_extraction(self,cycle_length=4, block_length=16, num_parallel_calls=-1):
        '''
        Generates a parallel consuming of items in the list of the original object given the reading function. For example reading files, or images.

        :param cycle_length: defaults to 4, read TF docs for more details.
        :param block_length: defaults to 16, read TF docs for more details.
        :param num_parallel_calls: defaults to Autotune, read TF docs for more details.
        @return:
        '''
        assert self.gen_object is not None, 'Generator must be specified to parallelize extraction'
        assert hasattr(self.gen_object, 'list'), f'Generation {type(self.gen_object)} has not attribute list'
        try:
            items_dataset = tf.data.Dataset.from_tensor_slices(self.gen_object.list)
        except TypeError as e:
            warn('Generate list maybe too complex to make parallelization')
            items_dataset = from_object(_Reader(lambda x: tuple(x), self.gen_object.list), training=self.training).build()
        types, shapes = _get_types_output(self.dataset.element_spec)
        if num_parallel_calls ==-1:
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        self.dataset = items_dataset.interleave(
                lambda x: tf.data.Dataset.from_generator(self.gen_object.read_fcn,
                                                     types,
                                                     shapes,
                                                     args=(x,)),
                num_parallel_calls=num_parallel_calls,
                cycle_length=cycle_length,
                block_length=block_length)
        return self

    def prefetch(self,buffer_size):
        '''Preloads samples on the tensor flow session i.e. memory to be processed.abs($0)

        :param buffer_size: size of the preloaded samples. If batch is specified then it loads a buffer_size of batches. For example, buffer_size=2 with batches of 100 will load 200 samples to the memory.
        :type buffer_size: int
        '''
        self.dataset = self.dataset.prefetch(buffer_size)
        return self
    def repeat(self,count=None):
        '''Creates a concatenated repeated dataset_builder

        :param count: Number of repreatitions
        :type count: int
        '''
        self.dataset = self.dataset.repeat(count=count)
        return self
    def shuffle(self,buffer_size,seed=None,reshuffle_each_iteration=None):
        self.dataset = self.dataset.shuffle(buffer_size,seed=seed,reshuffle_each_iteration=reshuffle_each_iteration)
        return self
    def build(self):
        '''Creates an augmented dataset that contains the arguments to be used in the method :class:`tf.keras.model.fit()`
        '''
        self._build_argments_fit()
        return self.dataset

class GeneratorBase():
    """Wraps an object with a method getitem to make it an iterable class

    :param obj: Instance of object that access data
    :type obj: :class:`object`
    :param getitem_fcn: Name of the method in the object to access data
    :type getitem_fcn: str
    :param length: length of the dataset, if None then infers from len() function, defaults to None
    :type length: int
    """
    def __init__(self,obj,getitem_fcn=None,itemlist_name=None,length=None):
        self.obj = obj
        if getitem_fcn is None:
            if hasattr(obj,'__getitem__'):
                self.getitem = obj.__getitem__
            else:
                raise Exception('Obj {} has no attribute {}'.format(type(obj),'__getitem__'))
        else:
            if hasattr(obj,getitem_fcn):
                self.getitem = getattr(obj,getitem_fcn)
            else:
                raise Exception('Obj {} has no attribute {}'.format(type(obj),getitem_fcn))
        # complete implementation of _Reader base class
        self.read_fcn = self.getitem

        if itemlist_name is None:
            if hasattr(obj,'list'):
                self.list = obj.list
            else:
                raise Exception('Obj {} has no attribute {}'.format(type(obj),'list'))
        else:
            if hasattr(obj,itemlist_name):
                self.list = getattr(obj,itemlist_name)
            else:
                raise Exception('Obj {} has no attribute {}'.format(type(obj),itemlist_name))

        if hasattr(obj,'__len__'):
            self.length = len(obj)
        elif length is not None:
            self.length = length
        else:
            raise Exception("length value is None, and object has not attribute to infer")
        print('current length is: ', self.length)
        self.cnt = 0
    def __len__(self):
        return self.length
    def __call__(self):
        self.cnt = 0
        return self
    def __iter__(self):
        self.cnt = 0
        return self
    def __next__(self):
        if self.cnt>=self.length:
            raise StopIteration
        else:
            return self.send(None)
    def send(self, ignored_arg):
        current_value = self.getitem(self.cnt)
        self.cnt +=1
        # tf.data.Dataset fails to parse list, therefore the values is transformed into a tuple
        if isinstance(current_value,list):
            current_value = tuple(current_value)
        return current_value

    def throw(self, type=None, value=None, traceback=None):
        """ Raise a :class:`StopIteration`
        """
        raise StopIteration



def from_function(read_fcn, list,training=True,undetermined_shape=None):
    obj = _Reader(read_fcn, list)
    return from_object(obj,training=training,undetermined_shape=undetermined_shape)

def from_object(obj,getitem_fcn=None,itemlist_name=None,training=True,undetermined_shape=None):
    """ Creates a tf.data.Dataset object with configuration parameters for fitting

    :param obj:  Object instance of the data with 'getitem_fcn' function to access dataset
    :param getitem_fcn: getitem_fcn Name of the method to access data . getitem_fcn can have any name defined for the in the class 'obj'. If not specified infers '__getitem__' as name of the access function
    :type getitem_fcn: str, optional
    :param itemlist_name: name of the list containing samples on the object, if None name is "list"
    :type itemlist_name: str, optional

    :param training: Specify training/validation flag
    :type training: bool, optional
    :param undetermined_shape: defines positions in the shape vector where dimensions are undetermided
    :type undetermined_shape: iterable, optional
    :return: An object :class:`tf.data.Dataset` from the obj dataset
    """
    gen = GeneratorBase(obj,getitem_fcn,itemlist_name)
    # infer output types
    value = gen.getitem(0)
    # extract value from generator
    is_generator = False
    if isinstance(value, types.GeneratorType):
        is_generator = True
        value = next(value)
    # Infers the data types and shapes:
    if isinstance(value,(tuple,list)):
        # checks if list or tuple object, multiple elements and not dictionary
        output_types = tuple(get_tf_dtype(v) for v in value)
        output_shapes = tuple(get_tf_shape(v) for v in value)
    elif isinstance(value,np.ndarray):
        # checks if single value
        output_types = (get_tf_dtype(value))
        output_shapes = (tf.TensorShape(list(value.shape)))
    elif isinstance(value,str):
        # checks if string
        output_types = (tf.string)
        output_shapes = (tf.TensorShape([]))
    elif isinstance(value,(int,float)):
        # checks if single value
        output_types = (get_tf_dtype(value))
        output_shapes = (tf.TensorShape([]))
    elif isinstance(value,dict):
        # checks if iterable and dictionary
        raise NotImplemented
        output_types = (tf.int32)
        output_shapes = (tf.TensorShape([]))
    else:
        raise Exception(f'Output types and shapes couldn\'t be identified. The input value {type(value)} has to be a single np.array or a list, tuple or dictionary of np.arrays')
    if undetermined_shape is not None:
        def apply_undetermined(output_shape,targets):
            # replaces target positions with None to configure undetermined dimension
            print('output_shape,targets', output_shape,targets)
            output_shape = list(output_shape)
            for target in targets:
                output_shape[target] = None
            return tuple(output_shape)
        if len(undetermined_shape)>0 and is_iterable(undetermined_shape[0]):
            output_shapes = tuple(map(lambda x: apply_undetermined(*x),zip(output_shapes,undetermined_shape)))
        else:
            output_shapes = apply_undetermined(output_shapes, undetermined_shape)
    if is_generator:

        if itemlist_name is not None:
            dataset = tf.data.Dataset.from_tensor_slices(getattr(obj,itemlist_name))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(obj.list)

        if getitem_fcn is not None:
            dataset = dataset.interleave(
                lambda x: tf.data.Dataset.from_generator(getattr(obj,getitem_fcn), output_types, output_shapes,
                                                         args=(x,)),num_parallel_calls=tf.data.experimental.AUTOTUNE)
            aug_dataset = AugmentedDataset(dataset, length=len(gen), training=training)
        else:
            read_fcn = obj.read_fcn if hasattr(obj, 'read_fcn') else obj.__getitem__
            dataset = dataset.interleave(
                lambda x: tf.data.Dataset.from_generator(read_fcn, output_types, output_shapes, args=(x,)),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            aug_dataset = AugmentedDataset(dataset, length=len(gen), training=training)

    else:
        dataset = tf.data.Dataset.from_generator(gen, output_types, output_shapes)
        if not hasattr(obj, 'read_fcn'):
            obj.read_fcn = obj.__getitem__ if getitem_fcn is None else getattr(obj, getitem_fcn)
        if not hasattr(obj, 'list'):
            assert itemlist_name is not None, 'itemlist_name has to be defined or set the list of values to self.list'
            obj.list = getattr(obj, itemlist_name)
        aug_dataset = AugmentedDataset(dataset, gen_object= obj, length=len(gen), training=training)
    return aug_dataset

