import numpy as np
import tensorflow as tf

def is_iterable(value):
    """ Verifies the value is an is_iterable

    :param value: value to identify if iterable or not.
    """
    try:
        iterable_obj = iter(value)
        return True
    except TypeError as te:
        return False
def get_tf_shape(value):
    """ Obtains the shape of an variable

    :param value: input value can be :class:`numpy.ndarray`, numeric or string class. It supports list of numerics or string but not nested lists.
    """
    if isinstance(value,(list,tuple)): # to support list of list or list of tuples
        return [len(list)]
    elif isinstance(value,np.ndarray):
        return list(value.shape)
    elif not is_iterable(value) or isinstance(value, str): # input can be a list of values or strings
        return []


def get_tf_dtype(value):
    """ Obtains tensorflow datatype

    The available transformations are:
        * tf.float16: 16-bit half-precision floating-point.
        * tf.float32: 32-bit single-precision floating-point.
        * tf.float64: 64-bit double-precision floating-point.
        * tf.complex64: 64-bit single-precision complex.
        * tf.complex128: 128-bit double-precision complex.
        * tf.int8: 8-bit signed integer.
        * tf.uint8: 8-bit unsigned integer.
        * tf.uint16: 16-bit unsigned integer.
        * tf.uint32: 32-bit unsigned integer.
        * tf.uint64: 64-bit unsigned integer.
        * tf.int16: 16-bit signed integer.
        * tf.int32: 32-bit signed integer.
        * tf.int64: 64-bit signed integer.
        * tf.bool: Boolean.
        * tf.string: String.

    :param value: Value to identify class from.
    """
    value = get_single_value(value)

    if isinstance(value,np.float16):
        return tf.float16 # 16-bit half-precision floating-point.
    elif isinstance(value,np.float32):
        return tf.float32   # 32-bit single-precision floating-point.
    elif isinstance(value,np.float64):
        return tf.float64   # 64-bit double-precision floating-point.
    elif isinstance(value,np.complex64):
        return tf.complex64   # 64-bit single-precision complex.
    elif isinstance(value,np.complex128):
        return tf.complex128   # 128-bit double-precision complex.
    elif isinstance(value,np.int8):
        return tf.int8   # 8-bit signed integer.
    elif isinstance(value,np.uint8):
        return tf.uint8   # 8-bit unsigned integer.
    elif isinstance(value,np.uint16):
        return tf.uint16   # 16-bit unsigned integer.
    elif isinstance(value,(np.uint32)):
        return tf.uint32   # 32-bit unsigned integer.
    elif isinstance(value,np.uint64):
        return tf.uint64   # 64-bit unsigned integer.
    elif isinstance(value,np.int16):
        return tf.int16   # 16-bit signed integer.
    elif isinstance(value,(int,np.int32)):
        return tf.int32   # 32-bit signed integer.
    elif isinstance(value,np.int64):
        return tf.int64   # 64-bit signed integer.
    elif isinstance(value,np.bool):
        return tf.bool   #Boolean.
    elif isinstance(value,str):
        return tf.string   #String.
    else:
        raise Exception("Type not found for value {}".format(type(value)))
def get_single_value(value,counter=0):
    """ Recursively tracks a single element of value. It assumes all elements as of the same kind

    :param value: Value get a single value from.
    """
    if is_iterable(value) and not isinstance(value, str):
        iterable = iter(value)
        return get_single_value(next(iterable), counter=counter+1)
    else:
        return value
