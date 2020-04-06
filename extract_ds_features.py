"""
    Script for extracting DeepSpeech features from audio file.
"""

import os
import argparse
from deepspeech_store import get_deepspeech_model_file
from deepspeech_features import conv_audios_to_deepspeech


def parse_args():
    """
    Create python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Extract DeepSpeech features from audio file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="path to input audio file or directory")
    parser.add_argument(
        "--output",
        type=str,
        help="path to output file with DeepSpeech features")
    parser.add_argument(
        "--deepspeech",
        type=str,
        help="path to DeepSpeech 0.1.0 frozen model")

    args = parser.parse_args()
    return args


def extract_features(in_audios,
                     out_files,
                     deepspeech_pb_path):
    """
    Real extract audio from video file.

    Parameters
    ----------
    in_audios : list of str
        Paths to input audio files.
    out_files : list of str
        Paths to output files with DeepSpeech features.
    deepspeech_pb_path : str
        Path to DeepSpeech 0.1.0 frozen model.
    """
    for i, in_audio in enumerate(in_audios):
        if not out_files[i]:
            file_stem, _ = os.path.splitext(in_audio)
            out_files[i] = file_stem + ".ds.npy"
    conv_audios_to_deepspeech(
        audios=in_audios,
        out_files=out_files,
        deepspeech_pb_path=deepspeech_pb_path)


def main():
    """
    Main body of script.
    """
    args = parse_args()
    in_audio = os.path.expanduser(args.input)
    if not os.path.exists(in_audio):
        raise Exception("Input file/directory doesn't exist: {}".format(in_audio))
    deepspeech_pb_path = args.deepspeech
    if deepspeech_pb_path is None:
        deepspeech_pb_path = ""
    if deepspeech_pb_path:
        deepspeech_pb_path = os.path.expanduser(args.deepspeech)
    if not os.path.exists(deepspeech_pb_path):
        deepspeech_pb_path = get_deepspeech_model_file()
    if os.path.isfile(in_audio):
        extract_features(
            in_audios=[in_audio],
            out_files=[args.output],
            deepspeech_pb_path=deepspeech_pb_path)
    else:
        audio_file_paths = []
        for file_name in os.listdir(in_audio):
            if not os.path.isfile(os.path.join(in_audio, file_name)):
                continue
            _, file_ext = os.path.splitext(file_name)
            if file_ext.lower() == ".wav":
                audio_file_path = os.path.join(in_audio, file_name)
                audio_file_paths.append(audio_file_path)
        out_file_paths = [""] * len(audio_file_paths)
        extract_features(
            in_audios=audio_file_paths,
            out_files=out_file_paths,
            deepspeech_pb_path=deepspeech_pb_path)


if __name__ == "__main__":
    main()
