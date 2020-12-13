try:
    import tensorflow as tf
except Exception as e:
    print('tensorflow import failed', e)
from .factories import from_function, AugmentedDataset
from .utils import get_read_fcn, create_label_dict, is_supported_format, get_video_length
from pathlib import Path
from warnings import warn
import numpy as np

def __get_undetermined_dims(x_type):
    if x_type == 'label':
        return []
    elif x_type == 'video':
        return [0,1,2]
    elif x_type == 'image':
        return [0,1]

def make_dataset(x_type, y_type, x_path=None, y_path=None, x_size=None, y_size=None, training=True, video_frames=None,
                 video_cropping=None, one_hot_encoding=False):
    """ Create custom dataset from a path list

    :param x_type: Defines the type of the input data to the model. It can be: label, video or image. The proper reading is generated accordingly.
    :param y_type: Defines the type of the target data to the model, idem as above
    :param x_path: Path to the dataset of inputs, the path is expected to contain images or videos sorted in a way that the name of the containing parent is the label, if label is relevant. For example cat/image1.png and dog/image2.png. All the files are indexed as individual samples.
    :param y_path: Path to the dataset of targets, idem as above.
    :param x_size: Size of the image or video for the input to the model
    :param y_size: Size of the image or video for the target to the model
    :param training: Specify training/validation flag
    :type training: bool, optional
    :param video_frames: number of frames of the output video if data type is video
    :param video_cropping: video cropping method creates a crop of the video with a length defined by video frames. Working modes are single and multi.  single where the video will be just from the first frame to the number of video_frames defined; or the multi where the video is cropped sequences of clips with the number of frames defined by video_frames.
    :param one_hot_encoding: Activate one hot encoding for the label input
    :return: Created dataset :class:`tf.data.Dataset` with the pairs input and target (x,y)
    """
    assert x_path is not None or y_path is not None, 'Neither x_path nor y_path was defined'
    if x_type in ['video', 'image']:
        assert x_size is not None, 'If x_type is video or image shape should be specified'
    if y_type in ['video', 'image']:
        assert y_size is not None, 'If y_type is video or image shape should be specified'
    if video_cropping is not None:
        assert x_type == 'video' and y_type == 'label', f"Video cropping not implemented for configuration {x_type} and {y_type}"
        assert video_cropping in ['multi', 'single'], 'Supported methods for the video cropping method is multi and ' \
                                                      'single '
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

    x_read_fcn = get_read_fcn(x_type, create_label_dict(x_path, one_hot_encoding=one_hot_encoding)) if x_type == 'label' else get_read_fcn(x_type)
    y_read_fcn = get_read_fcn(y_type, create_label_dict(y_path, one_hot_encoding=one_hot_encoding)) if y_type == 'label' else get_read_fcn(x_type)
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
    if video_cropping is None:
        return from_function(read_fcn, paths, undetermined_shape=undetermined, training=training).map(resize_inputs)
    elif video_cropping == 'single':
        # compute video length when not specified
        if video_frames is None:
            lengths = from_function(get_video_length, [p[0] for p in paths]).map(lambda x: tf.dtypes.cast(x, tf.float32)).build()
            video_frames = int(lengths.reduce(np.inf, tf.math.minimum).numpy())
        # create the dataset
        def read_fcn_crop(xy_path):
            x, y = read_fcn(xy_path)
            x = x[0:video_frames]
            return x, y
        return from_function(read_fcn_crop, paths, undetermined_shape=([1,2],[]), training=training).map(resize_inputs)
    elif video_cropping == 'multi':
        # compute video length when not specified
        lengths = list(from_function(get_video_length, [p[0] for p in paths]).build().as_numpy_iterator())
        lengths = [str(int(l)) for l in lengths]
        if video_frames is None:
            video_frames = int(min(lengths))
        # build a zipped path and lengths crop function
        def crop(composed_path):
            if isinstance(composed_path, str): # , f"Composed path must be an <str: String class> class. It is : {composed_path.__class__}"
                items = composed_path.split('-->')
                xy_path, length = (items[0], items[1]), int(items[2])
            else:
                items = tf.strings.split(composed_path, sep='-->')
                def substr(s, init, end):
                    start = s.find(init) + len(end)
                    end = s.find(end)
                    return s[start:end]
                path1 = substr(str(items[0]), "b'", "',")
                path2 = substr(str(items[1]), "b'", "',")
                xy_path, length = (path1, path2), int(items[2])
            X, y = read_fcn(xy_path)
            N = int(length // video_frames)
            Xs = [X[i * video_frames:(i + 1) * video_frames] for i in range(N)]
            ys = len(Xs)*[y]
            for Xi,yi in zip(Xs,ys):
                yield Xi, yi

        composed_paths = ['-->'.join([p[0],p[1],l]) for p, l in zip(paths, lengths)]
        return from_function(crop, composed_paths, undetermined_shape=([1,2],[]), training=training).map(resize_inputs)


