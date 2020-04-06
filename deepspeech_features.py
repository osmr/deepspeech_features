"""
    DeepSpeech features processing routines.
"""

__all__ = ['conv_audios_to_deepspeech']

import numpy as np
import resampy
from scipy.io import wavfile
from python_speech_features import mfcc
import tensorflow as tf


def conv_audios_to_deepspeech(audios,
                              out_files,
                              deepspeech_pb_path,
                              audio_window_size=16,
                              audio_window_stride=1):
    graph, logits_ph, input_node_ph, input_lengths_ph = prepare_deepspeech_net(deepspeech_pb_path)

    with tf.compat.v1.Session(graph=graph) as sess:
        for audio_file_path, out_file_path in zip(audios, out_files):
            audio_sample_rate, audio = wavfile.read(audio_file_path)
            ds_features = pure_conv_audio_to_deepspeech(
                audio=audio,
                audio_sample_rate=audio_sample_rate,
                audio_window_size=audio_window_size,
                audio_window_stride=audio_window_stride,
                net_fn=lambda x: sess.run(
                    logits_ph,
                    feed_dict={
                        input_node_ph: x[np.newaxis, ...],
                        input_lengths_ph: [x.shape[0]]}))
            np.save(out_file_path, ds_features)


def prepare_deepspeech_net(deepspeech_pb_path):
    # Load graph and place_holders:
    with tf.io.gfile.GFile(deepspeech_pb_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.compat.v1.get_default_graph()
    tf.import_graph_def(graph_def, name="deepspeech")
    logits_ph = graph.get_tensor_by_name("deepspeech/logits:0")
    input_node_ph = graph.get_tensor_by_name("deepspeech/input_node:0")
    input_lengths_ph = graph.get_tensor_by_name("deepspeech/input_lengths:0")

    return graph, logits_ph, input_node_ph, input_lengths_ph


def pure_conv_audio_to_deepspeech(audio,
                                  audio_sample_rate,
                                  audio_window_size,
                                  audio_window_stride,
                                  net_fn):
    target_sample_rate = 16000
    resampled_audio = resampy.resample(
        x=audio.astype(np.float),
        sr_orig=audio_sample_rate,
        sr_new=target_sample_rate)
    input_vector = conv_audio_to_deepspeech_input_vector(
        audio=resampled_audio.astype(np.int16),
        sample_rate=target_sample_rate,
        num_cepstrum=26,
        num_context=9)

    network_output = net_fn(input_vector)

    deepspeech_fps = 50
    video_fps = 60
    if video_fps != deepspeech_fps:
        audio_len_s = float(audio.shape[0]) / audio_sample_rate
        num_frames = int(round(audio_len_s * video_fps))
        network_output = interpolate_features(
            features=network_output[:, 0],
            input_rate=deepspeech_fps,
            output_rate=video_fps,
            output_len=num_frames)

    # Make windows:
    zero_pad = np.zeros((int(audio_window_size / 2), network_output.shape[1]))
    network_output = np.concatenate((zero_pad, network_output, zero_pad), axis=0)
    windows = []
    for window_index in range(0, network_output.shape[0] - audio_window_size, audio_window_stride):
        windows.append(network_output[window_index:window_index + audio_window_size])

    return np.array(windows)


def conv_audio_to_deepspeech_input_vector(audio,
                                          sample_rate,
                                          num_cepstrum,
                                          num_context):
    # Get mfcc coefficients:
    features = mfcc(
        signal=audio,
        samplerate=sample_rate,
        numcep=num_cepstrum)

    # We only keep every second feature (BiRNN stride = 2):
    features = features[::2]

    # One stride per time step in the input:
    num_strides = len(features)

    # Add empty initial and final contexts:
    empty_context = np.zeros((num_context, num_cepstrum), dtype=features.dtype)
    features = np.concatenate((empty_context, features, empty_context))

    # Create a view into the array with overlapping strides of size
    # numcontext (past) + 1 (present) + numcontext (future):
    window_size = 2 * num_context + 1
    train_inputs = np.lib.stride_tricks.as_strided(
        features,
        shape=(num_strides, window_size, num_cepstrum),
        strides=(features.strides[0], features.strides[0], features.strides[1]),
        writeable=False)

    # Flatten the second and third dimensions:
    train_inputs = np.reshape(train_inputs, [num_strides, -1])

    train_inputs = np.copy(train_inputs)
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

    return train_inputs


def interpolate_features(features,
                         input_rate,
                         output_rate,
                         output_len):
    input_len = features.shape[0]
    num_features = features.shape[1]
    input_timestamps = np.arange(input_len) / float(input_rate)
    output_timestamps = np.arange(output_len) / float(output_rate)
    output_features = np.zeros((output_len, num_features))
    for feature_idx in range(num_features):
        output_features[:, feature_idx] = np.interp(
            x=output_timestamps,
            xp=input_timestamps,
            fp=features[:, feature_idx])
    return output_features
