from collections.abc import Generator
import tensorflow as tf
from utils import get_tf_dtype, get_tf_shape, is_iterable
import numpy as np

class GeneratorBase(Generator):
    '''
    Wraps an object with a method getitem to make it an iterable class
    :param obj: instance of object that access data
    :type arg: str
    :param getitem_fcn: The variable arguments are used for ...
    :type arg: str
    :param `**kwargs`: The keyword arguments are used for ...
    :ivar arg: This is where we store arg
    :vartype arg: str

    '''
    def __init__(self,obj,getitem_fcn=None,lenght=None):
        """ inits Spamfilter with training data

        :param training_dir: path of training directory with subdirectories
         '/ham' and '/spam'
        """
        self.obj = obj
        if getitem_fcn is None:
            if hasattr(obj,'__getitem__'):
                self.getitem = getattr(obj,'__getitem__')
            else:
                raise Exception('Obj {} has no attribute {}'.format(type(obj),'__getitem__'))
        else:
            if hasattr(obj,getitem_fcn):
                self.getitem = getattr(obj,getitem_fcn)
            else:
                raise Exception('Obj {} has no attribute {}'.format(type(obj),getitem_fcn))


        if hasattr(obj,'__len__'):
            self.lenght = len(obj)
        elif length is not None:
            self.length = lenght
        else:
            raise ValueException("Lenght value is None, and object has not attribute to infer")
        print('current lenght is: ', self.lenght)
        self.cnt = 0
    def __call__(self):
        return self
    def __next__(self):
        """ Generates token frequency table from training emails
        :return:  dict{k,v}:  spam/ham frequencies
        k = (str)token, v = {spam_freq: , ham_freq:, prob_spam:, prob_ham:}
        """
        if self.cnt>=self.lenght:
            raise StopIteration
        else:
            return self.send(None)
    def send(self, ignored_arg):
        """ Generates token frequency table from training emails
        :param: ignored_arg must be None
        :return:  dict{k,v}:  spam/ham frequencies
        k = (str)token, v = {spam_freq: , ham_freq:, prob_spam:, prob_ham:}
        """
        current_value = self.getitem(self.cnt)
        self.cnt +=1
        return current_value

    def throw(self, type=None, value=None, traceback=None):
        """ Generates token frequency table from training emails
        :return:  dict{k,v}:  spam/ham frequencies
        k = (str)token, v = {spam_freq: , ham_freq:, prob_spam:, prob_ham:}
        """
        raise StopIteration


def from_object(obj,getitem_fcn=None):
    """ Creates a tf.data.Dataset object with configuration parameters for fitting
    :param obj:  Object instance of the data with 'getitem_fcn' function to access dataset
    :param getitem_fcn: getitem_fcn Name of the method to access data . getitem_fcn can have any name defined for the in the class 'obj'. If not specified infers '__getitem__' as name of the access function
    :return: An object :class:`tf.data.Dataset` from the obj dataset
    """
    gen = GeneratorBase(obj,getitem_fcn)
    # infer output types
    print('---------------------->>>>')
    print('type(gen.getitem(0)) = ',type(gen.getitem(0)), 'double type: ', type(type(gen.getitem(0))))
    print('---------------------->>>>')
    value = gen.getitem(0)
    # Infers the data types and shapes:
    if isinstance(value,(tuple,list)):
        # checks if list or tuple object, multiple elements and not dictionary
        output_types = tuple(get_tf_dtype(v) for v in value)
        output_shapes = tuple(get_tf_shape(v) for v in value)
    elif isinstance(value,np.ndarray):
        # checks if single value
        output_types = (get_tf_dtype(value))
        output_shapes = (tf.TensorShape(list(value.shape)))
    elif isinstance(value,dict):
        # checks if iterable and dictionary
        print('not implement now dictionary')
        output_types = (tf.int32)
        output_shapes = (tf.TensorShape([]))
    else:
        raise ValueException(f'The input value {type(value)} has to be a single np.array or a list, tuple or dictionary of np.arrays')

    a = gen()
    print('----a callable function', a, type(a))
    print('------>>>>>>> output_types itentified...:', output_types)
    return tf.data.Dataset.from_generator(gen, output_types, output_shapes)



if __name__ == '__main__':
    # c = Counter()
    # dataset = from_object(c,'read')
    # print(list(dataset))
    print('MAIN: ')
