import os, sys
import numpy as np
try:
    import tensorflow as tf
except Exception as e:
    print('tensorflow import failed', e)
from PIL import Image
import numpy
from skvideo.io import vread, ffprobe
from warnings import warn



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
    if isinstance(value, (list, tuple)):  # to support list of list or list of tuples
        return [len(value)]
    elif isinstance(value, np.ndarray):
        return list(value.shape)
    elif not is_iterable(value) or isinstance(value, str):  # input can be a list of values or strings
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

    if isinstance(value, np.float16):
        return tf.float16  # 16-bit half-precision floating-point.
    elif isinstance(value, np.float32):
        return tf.float32  # 32-bit single-precision floating-point.
    elif isinstance(value, (np.float64, np.float)):
        return tf.float64  # 64-bit double-precision floating-point.
    elif isinstance(value, np.complex64):
        return tf.complex64  # 64-bit single-precision complex.
    elif isinstance(value, np.complex128):
        return tf.complex128  # 128-bit double-precision complex.
    elif isinstance(value, np.int8):
        return tf.int8  # 8-bit signed integer.
    elif isinstance(value, np.uint8):
        return tf.uint8  # 8-bit unsigned integer.
    elif isinstance(value, np.uint16):
        return tf.uint16  # 16-bit unsigned integer.
    elif isinstance(value, (np.uint32)):
        return tf.uint32  # 32-bit unsigned integer.
    elif isinstance(value, np.uint64):
        return tf.uint64  # 64-bit unsigned integer.
    elif isinstance(value, np.int16):
        return tf.int16  # 16-bit signed integer.
    elif isinstance(value, (int, np.int32)):
        return tf.int32  # 32-bit signed integer.
    elif isinstance(value, np.int64):
        return tf.int64  # 64-bit signed integer.
    elif isinstance(value, np.bool):
        return tf.bool  # Boolean.
    elif isinstance(value, str):
        return tf.string  # String.
    elif isinstance(value, int):
        return tf.int32
    elif isinstance(value, float):
        return tf.float64
    else:
        raise Exception("Type not found for value {}".format(type(value)))


def get_single_value(value, counter=0):
    """ Recursively tracks a single element of value. It assumes all elements as of the same kind

    :param value: Value get a single value from.
    """
    if is_iterable(value) and not isinstance(value, str):
        iterable = iter(value)
        return get_single_value(next(iterable), counter=counter + 1)
    elif isinstance(value, np.ndarray):
        return value.item()
    else:
        return value


def get_parent_path(path):
    try:
        path = os.path.abspath(str(path))
        if path.find(os.sep)>-1:
            return path.split(os.sep)[-2]
        elif path.find('/') > -1:
            return path.split('/')[-2]
        elif path.find('\\') > -1:
            return path.split('\\')[-2]
        else:
            return path

    except Exception:
        t, v, tb = sys.exc_info()
        raise ValueError(f'Failed to infer parent folder of path {path}').with_traceback(tb)


def create_label_dict(paths, one_hot_encoding=False):
    """Creates a label dictionaty from list of paths

    :param paths: Path from where the label dictionary is created. The parent folder is considered as label for the images.
    :type paths: list
    :return: Dictionary of labels sorted alphabetically
    :rtype: dict
    """
    labels = set(sorted(map(get_parent_path, paths)))
    if one_hot_encoding:
        def encode(i):
            u = np.zeros(len(labels))
            u[i] = 1
            return u
        return {label: encode(i) for i, label in enumerate(labels)}
    else:
        return {label: i for i, label in enumerate(labels)}


def get_read_fcn(data_type, label_dict=None):
    def read_image(path):
        i = Image.open(path)
        a = numpy.asarray(i)  # a is readonly
        return np.array(a)

    def read_label(path, label_dict):
        label = get_parent_path(path)
        return label_dict[label]

    def read_video(path):
        return vread(path)

    if data_type == 'image':
        read_fcn = read_image
    elif data_type == 'video':
        read_fcn = read_video
    elif data_type == 'label':
        assert label_dict is not None, f'label_dict is not specified. This label dict should be a dictionary that ' \
                                       f'translate labels to codes '
        read_fcn = lambda x: read_label(x, label_dict)
    else:
        raise Exception(f' Data type \'{data_type}\' is not supported. Use image, video or label')
    return read_fcn


def is_supported_format(filename):
    is_supported = filename.endswith('.bmp') or filename.endswith('.eps') or filename.endswith(
        '.gif') or filename.endswith('.icns') or filename.endswith('.ico') or filename.endswith(
        '.im') or filename.endswith('.jpg')  or filename.endswith('.jpeg') or filename.endswith(
        '.msp') or filename.endswith(
        '.pcx') or filename.endswith('.png') or filename.endswith('.ppm') or filename.endswith(
        '.sgi') or filename.endswith('.tiff') or filename.endswith('.webp') or filename.endswith(
        '.xbm') or filename.endswith('.3dostr') or filename.endswith('.3g2') or filename.endswith(
        '.3gp') or filename.endswith('.4xm') or filename.endswith('.a64') or filename.endswith(
        '.aa') or filename.endswith('.aac') or filename.endswith('.ac3') or filename.endswith(
        '.acm') or filename.endswith('.act') or filename.endswith('.adf') or filename.endswith(
        '.adp') or filename.endswith('.ads') or filename.endswith('.adts') or filename.endswith(
        '.adx') or filename.endswith('.aea') or filename.endswith('.afc') or filename.endswith(
        '.aiff') or filename.endswith('.aix') or filename.endswith('.alaw') or filename.endswith(
        '.alias_pix') or filename.endswith('.amr') or filename.endswith('.amrnb') or filename.endswith(
        '.amrwb') or filename.endswith('.anm') or filename.endswith('.apc') or filename.endswith(
        '.ape') or filename.endswith('.apng') or filename.endswith('.aptx') or filename.endswith(
        '.aptx_hd') or filename.endswith('.aqtitle') or filename.endswith('.argo_asf') or filename.endswith(
        '.asf') or filename.endswith('.asf_o') or filename.endswith('.asf_stream') or filename.endswith(
        '.ass') or filename.endswith('.ast') or filename.endswith('.au') or filename.endswith(
        '.av1') or filename.endswith('.avi') or filename.endswith('.avisynth') or filename.endswith(
        '.avm2') or filename.endswith('.avr') or filename.endswith('.avs') or filename.endswith(
        '.avs2') or filename.endswith('.bethsoftvid') or filename.endswith('.bfi') or filename.endswith(
        '.bfstm') or filename.endswith('.bin') or filename.endswith('.bink') or filename.endswith(
        '.bit') or filename.endswith('.bmp_pipe') or filename.endswith('.bmv') or filename.endswith(
        '.boa') or filename.endswith('.brender_pix') or filename.endswith('.brstm') or filename.endswith(
        '.c93') or filename.endswith('.caf') or filename.endswith('.cavsvideo') or filename.endswith(
        '.cdg') or filename.endswith('.cdxl') or filename.endswith('.cine') or filename.endswith(
        '.codec2') or filename.endswith('.codec2raw') or filename.endswith('.concat') or filename.endswith(
        '.crc') or filename.endswith('.dash') or filename.endswith('.data') or filename.endswith(
        '.daud') or filename.endswith('.dcstr') or filename.endswith('.dds_pipe') or filename.endswith(
        '.dfa') or filename.endswith('.dhav') or filename.endswith('.dirac') or filename.endswith(
        '.dnxhd') or filename.endswith('.dpx_pipe') or filename.endswith('.dsf') or filename.endswith(
        '.dshow') or filename.endswith('.dsicin') or filename.endswith('.dss') or filename.endswith(
        '.dts') or filename.endswith('.dtshd') or filename.endswith('.dv') or filename.endswith(
        '.dvbsub') or filename.endswith('.dvbtxt') or filename.endswith('.dvd') or filename.endswith(
        '.dxa') or filename.endswith('.ea') or filename.endswith('.ea_cdata') or filename.endswith(
        '.eac3') or filename.endswith('.epaf') or filename.endswith('.exr_pipe') or filename.endswith(
        '.f32be') or filename.endswith('.f32le') or filename.endswith('.f4v') or filename.endswith(
        '.f64be') or filename.endswith('.f64le') or filename.endswith('.ffmetadata') or filename.endswith(
        '.fifo') or filename.endswith('.fifo_test') or filename.endswith('.film_cpk') or filename.endswith(
        '.filmstrip') or filename.endswith('.fits') or filename.endswith('.flac') or filename.endswith(
        '.flic') or filename.endswith('.flv') or filename.endswith('.framecrc') or filename.endswith(
        '.framehash') or filename.endswith('.framemd5') or filename.endswith('.frm') or filename.endswith(
        '.fsb') or filename.endswith('.g722') or filename.endswith('.g723_1') or filename.endswith(
        '.g726') or filename.endswith('.g726le') or filename.endswith('.g729') or filename.endswith(
        '.gdigrab') or filename.endswith('.gdv') or filename.endswith('.genh') or filename.endswith(
        '.gif') or filename.endswith('.gif_pipe') or filename.endswith('.gsm') or filename.endswith(
        '.gxf') or filename.endswith('.h261') or filename.endswith('.h263') or filename.endswith(
        '.h264') or filename.endswith('.hash') or filename.endswith('.hcom') or filename.endswith(
        '.hds') or filename.endswith('.hevc') or filename.endswith('.hls') or filename.endswith(
        '.hnm') or filename.endswith('.ico') or filename.endswith('.idcin') or filename.endswith(
        '.idf') or filename.endswith('.iff') or filename.endswith('.ifv') or filename.endswith(
        '.ilbc') or filename.endswith('.image2') or filename.endswith('.image2pipe') or filename.endswith(
        '.ingenient') or filename.endswith('.ipmovie') or filename.endswith('.ipod') or filename.endswith(
        '.ircam') or filename.endswith('.ismv') or filename.endswith('.iss') or filename.endswith(
        '.iv8') or filename.endswith('.ivf') or filename.endswith('.ivr') or filename.endswith(
        '.j2k_pipe') or filename.endswith('.jacosub') or filename.endswith('.jpeg_pipe') or filename.endswith(
        '.jpegls_pipe') or filename.endswith('.jv') or filename.endswith('.kux') or filename.endswith(
        '.latm') or filename.endswith('.lavfi') or filename.endswith('.libopenmpt') or filename.endswith(
        '.live_flv') or filename.endswith('.lmlm4') or filename.endswith('.loas') or filename.endswith(
        '.lrc') or filename.endswith('.lvf') or filename.endswith('.lxf') or filename.endswith(
        '.m4v') or filename.endswith('.matroska') or filename.endswith('.matroska') or filename.endswith(
        '.webm') or filename.endswith('.md5') or filename.endswith('.mgsts') or filename.endswith(
        '.microdvd') or filename.endswith('.mjpeg') or filename.endswith('.mjpeg_2000') or filename.endswith(
        '.mkvtimestamp_v2') or filename.endswith('.mlp') or filename.endswith('.mlv') or filename.endswith(
        '.mm') or filename.endswith('.mmf') or filename.endswith('.mov') or filename.endswith(
        '.mov') or filename.endswith('.mp4') or filename.endswith('.m4a') or filename.endswith(
        '.3gp') or filename.endswith('.3g2') or filename.endswith('.mj2') or filename.endswith(
        '.mp2') or filename.endswith('.mp3') or filename.endswith('.mp4') or filename.endswith(
        '.mpc') or filename.endswith('.mpc8') or filename.endswith('.mpeg') or filename.endswith(
        '.mpeg1video') or filename.endswith('.mpeg2video') or filename.endswith('.mpegts') or filename.endswith(
        '.mpegtsraw') or filename.endswith('.mpegvideo') or filename.endswith('.mpjpeg') or filename.endswith(
        '.mpl2') or filename.endswith('.mpsub') or filename.endswith('.msf') or filename.endswith(
        '.msnwctcp') or filename.endswith('.mtaf') or filename.endswith('.mtv') or filename.endswith(
        '.mulaw') or filename.endswith('.musx') or filename.endswith('.mv') or filename.endswith(
        '.mvi') or filename.endswith('.mxf') or filename.endswith('.mxf_d10') or filename.endswith(
        '.mxf_opatom') or filename.endswith('.mxg') or filename.endswith('.nc') or filename.endswith(
        '.nistsphere') or filename.endswith('.nsp') or filename.endswith('.nsv') or filename.endswith(
        '.null') or filename.endswith('.nut') or filename.endswith('.nuv') or filename.endswith(
        '.oga') or filename.endswith('.ogg') or filename.endswith('.ogv') or filename.endswith(
        '.oma') or filename.endswith('.opus') or filename.endswith('.paf') or filename.endswith(
        '.pam_pipe') or filename.endswith('.pbm_pipe') or filename.endswith('.pcx_pipe') or filename.endswith(
        '.pgm_pipe') or filename.endswith('.pgmyuv_pipe') or filename.endswith('.pictor_pipe') or filename.endswith(
        '.pjs') or filename.endswith('.pmp') or filename.endswith('.png_pipe') or filename.endswith(
        '.ppm_pipe') or filename.endswith('.psd_pipe') or filename.endswith('.psp') or filename.endswith(
        '.psxstr') or filename.endswith('.pva') or filename.endswith('.pvf') or filename.endswith(
        '.qcp') or filename.endswith('.qdraw_pipe') or filename.endswith('.r3d') or filename.endswith(
        '.rawvideo') or filename.endswith('.realtext') or filename.endswith('.redspark') or filename.endswith(
        '.rl2') or filename.endswith('.rm') or filename.endswith('.roq') or filename.endswith(
        '.rpl') or filename.endswith('.rsd') or filename.endswith('.rso') or filename.endswith(
        '.rtp') or filename.endswith('.rtp_mpegts') or filename.endswith('.rtsp') or filename.endswith(
        '.s16be') or filename.endswith('.s16le') or filename.endswith('.s24be') or filename.endswith(
        '.s24le') or filename.endswith('.s32be') or filename.endswith('.s32le') or filename.endswith(
        '.s337m') or filename.endswith('.s8') or filename.endswith('.sami') or filename.endswith(
        '.sap') or filename.endswith('.sbc') or filename.endswith('.sbg') or filename.endswith(
        '.scc') or filename.endswith('.sdl') or filename.endswith('.sdl2') or filename.endswith(
        '.sdp') or filename.endswith('.sdr2') or filename.endswith('.sds') or filename.endswith(
        '.sdx') or filename.endswith('.segment') or filename.endswith('.ser') or filename.endswith(
        '.sgi_pipe') or filename.endswith('.shn') or filename.endswith('.siff') or filename.endswith(
        '.singlejpeg') or filename.endswith('.sln') or filename.endswith('.smjpeg') or filename.endswith(
        '.smk') or filename.endswith('.smoothstreaming') or filename.endswith('.smush') or filename.endswith(
        '.sol') or filename.endswith('.sox') or filename.endswith('.spdif') or filename.endswith(
        '.spx') or filename.endswith('.srt') or filename.endswith('.stl') or filename.endswith(
        '.stream_segment') or filename.endswith('.ssegment') or filename.endswith('.streamhash') or filename.endswith(
        '.subviewer') or filename.endswith('.subviewer1') or filename.endswith('.sunrast_pipe') or filename.endswith(
        '.sup') or filename.endswith('.svag') or filename.endswith('.svcd') or filename.endswith(
        '.svg_pipe') or filename.endswith('.swf') or filename.endswith('.tak') or filename.endswith(
        '.tedcaptions') or filename.endswith('.tee') or filename.endswith('.thp') or filename.endswith(
        '.tiertexseq') or filename.endswith('.tiff_pipe') or filename.endswith('.tmv') or filename.endswith(
        '.truehd') or filename.endswith('.tta') or filename.endswith('.tty') or filename.endswith(
        '.txd') or filename.endswith('.ty') or filename.endswith('.u16be') or filename.endswith(
        '.u16le') or filename.endswith('.u24be') or filename.endswith('.u24le') or filename.endswith(
        '.u32be') or filename.endswith('.u32le') or filename.endswith('.u8') or filename.endswith(
        '.uncodedframecrc') or filename.endswith('.v210') or filename.endswith('.v210x') or filename.endswith(
        '.vag') or filename.endswith('.vc1') or filename.endswith('.vc1test') or filename.endswith(
        '.vcd') or filename.endswith('.vfwcap') or filename.endswith('.vidc') or filename.endswith(
        '.vividas') or filename.endswith('.vivo') or filename.endswith('.vmd') or filename.endswith(
        '.vob') or filename.endswith('.vobsub') or filename.endswith('.voc') or filename.endswith(
        '.vpk') or filename.endswith('.vplayer') or filename.endswith('.vqf') or filename.endswith(
        '.w64') or filename.endswith('.wav') or filename.endswith('.wc3movie') or filename.endswith(
        '.webm') or filename.endswith('.webm_chunk') or filename.endswith('.webm_dash_manifest') or filename.endswith(
        '.webp') or filename.endswith('.webp_pipe') or filename.endswith('.webvtt') or filename.endswith(
        '.wsaud') or filename.endswith('.wsd') or filename.endswith('.wsvqa') or filename.endswith(
        '.wtv') or filename.endswith('.wv') or filename.endswith('.wve') or filename.endswith(
        '.xa') or filename.endswith('.xbin') or filename.endswith('.xmv') or filename.endswith(
        '.xpm_pipe') or filename.endswith('.xvag') or filename.endswith('.xwd_pipe') or filename.endswith(
        '.xwma') or filename.endswith('.yop') or filename.endswith('.yuv4mpegpipe')
    return is_supported


def get_video_length(path):
    """
    Reads the number of frames from the metadata of a video file

    :param path: path to the video file .
    :return: Number of frames extracted with ffprobe.
    :rtype: float64
    """
    try:
        metadata = ffprobe(path)
        nb = 1.0 * int(metadata["video"]['@nb_frames'])
        return nb
    except Exception as e:
        if metadata == {}:
            warn('Error maybe come from parallelization. ffmpeg already includes multithreading. ffmpeg can be '
                 'compiled also with graphics accelerators as GPU.')
        raise e
