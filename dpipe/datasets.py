import tensorflow as tf
from .factories import from_function
from .utils import get_read_fcn, create_label_dict, is_supported_format
from pathlib import Path
from warnings import warn


def __get_undetermined_dims(x_type):
    if x_type == 'label':
        return []
    elif x_type == 'video':
        return [0,1,2]
    elif x_type == 'image':
        return [0,1]


def make_dataset(x_type, y_type,x_path=None, y_path=None,x_size=None,y_size=None):
    """ Create custom dataset from a path list

    :param x_type: Defines the type of the input data to the model. It can be: label, video or image. The proper reading is generated accordingly.
    :param y_type: Defines the type of the target data to the model, idem as above
    :param x_path: Path to the dataset of inputs, the path is expected to contain images or videos sorted in a way that the name of the containing parent is the label, if label is relevant. For example cat/image1.png and dog/image2.png. All the files are indexed as individual samples.
    :param y_path: Path to the dataset of targets, idem as above.
    :param x_size: Size of the image or video for the input to the model
    :param y_size: Size of the image or video for the target to the model
    :return: Created dataset :class:`tf.data.Dataset` with the pairs input and target (x,y)
    """
    assert x_path is not None or y_path is not None, 'Neither x_path nor y_path was defined'
    if x_type in ['video', 'image']:
        assert x_size is not None, 'If x_type is video or image shape should be specified'
    if y_type in ['video', 'image']:
        assert y_size is not None, 'If y_type is video or image shape should be specified'

    def to_list(path):
        if isinstance(path,list):
            return path
        else:
            paths = list(map(lambda x: x.as_posix(),filter(lambda x: x.is_file(), Path(path).rglob('*'))))
            paths = list(filter(is_supported_format, paths))
            assert len(paths)>0, f'Path {path} is empty.'
            return paths

    if not x_path is None and y_path is None:
        x_path = to_list(x_path)
        y_path = x_path
    elif x_path is None and not y_path is None:
        y_path = to_list(y_path)
        x_path = y_path
    else:
        x_path = to_list(x_path)
        y_path = to_list(y_path)
        if not x_path == y_path:
            warn('Input and Target path x_path and y_path, respectively, don not match exactly. Please consider align the data source paths.', RuntimeWarning)

    x_read_fcn = get_read_fcn(x_type, create_label_dict(x_path)) if x_type == 'label' else get_read_fcn(x_type)
    y_read_fcn = get_read_fcn(y_type, create_label_dict(y_path)) if y_type == 'label' else get_read_fcn(x_type)
    paths = list(zip(x_path,y_path))

    def read_fcn(xy_paths):
        x_path, y_path = xy_paths[0], xy_paths[1]
        return x_read_fcn(x_path), y_read_fcn(x_path)

    @tf.function
    def resize_inputs(x, y):
        if not x_type == 'label':
            x = tf.image.resize(x, x_size)
        if not y_type == 'label':
            y = tf.image.resize(y, y_size)
        return x, y
    undetermined = (__get_undetermined_dims(x_type), __get_undetermined_dims(y_type))
    return from_function(read_fcn, paths, undetermined_shape=undetermined).map(resize_inputs)

