import glob
import librosa
import librosa.display as ld
import librosa.feature as lf
import librosa.onset as lo
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import tensorflow as tf
import tensorflow_io as tfio
import uuid
import wave
import werkzeug.utils as wu


SAMPLE_RATE = 16000
AMR_EXTENSION = '.amr'
WAV_EXTENSION = '.wav'


def replace_ext_to_txt(name: str) -> str:
    """Replace the file extension to .txt
    :param name: original file name
    :return: file name whose extension replaced with .txt
    """
    return name.replace(name.rsplit('.', 1)[1], 'txt')


def get_amr_audio(url: str, file_name: str, file_dir: str) -> str:
    """Get AMR audio file from url
    :param url: the url redirecting to audio file storage
    :param file_name: the audio file name (assuming without extension)
    :param file_dir: the directory to save audio file
    :return: audio file path
    """
    if file_name and not file_name.endswith(AMR_EXTENSION):
        file_name += AMR_EXTENSION
    file_path = os.path.join(file_dir, file_name)
    response = requests.get(url, stream=True, timeout=60)
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)
    return file_path


def convert_amr_to_wav(amr_file_path: str) -> str:
    """Convert AMR audio file to WAV
    :param amr_file_path: AMR audio file path
    :return: WAV audio file path
    """
    wav_file_path = amr_file_path.replace(AMR_EXTENSION, WAV_EXTENSION)
    ffmpeg_command = 'ffmpeg -hide_banner -loglevel error'
    os.system(f'{ffmpeg_command} -y -i {amr_file_path} {wav_file_path}')
    return wav_file_path


def get_wav_from_urls(urls: [str], file_dir: str, tmp_dir: str) -> str:
    """Get WAV audio file from a list of AMR urls
    :param urls: the list of AMR urls redirecting to audio file storage
    :param file_dir: the directory to save reproduced audio file
    :param tmp_dir: the directory to save temporary audio file
    :return: WAV audio file path
    """
    amr_files = [url.rsplit('/', 1)[-1] for url in urls]
    amr_files = [wu.secure_filename(f) for f in amr_files]
    wav_file_paths = []
    for (url, amr_file) in zip(urls, amr_files):
        if amr_file.endswith(AMR_EXTENSION):
            wav_file = amr_file.replace(AMR_EXTENSION, WAV_EXTENSION)
        else:
            wav_file = amr_file + WAV_EXTENSION
        wav_file_path = os.path.join(tmp_dir, wav_file)
        if os.path.exists(wav_file_path):
            wav_file_paths.append(wav_file_path)
        else:
            try:
                amr_file_path = get_amr_audio(url, amr_file, tmp_dir)
                wav_file_paths.append(convert_amr_to_wav(amr_file_path))
                os.remove(amr_file_path)
            except requests.RequestException:
                continue
    wav_data = []
    for wav_file_path in wav_file_paths:
        with wave.open(wav_file_path, 'rb') as w:
            wav_data.append([w.getparams(), w.readframes(w.getnframes())])
    wav_file_path = wav_file_paths[0].replace(tmp_dir, file_dir)
    with wave.open(wav_file_path, 'wb') as w:
        w.setparams(wav_data[0][0])
        for i in range(len(wav_data)):
            w.writeframes(wav_data[i][1])
    return wav_file_path


def get_duration(file_path: str) -> float:
    """Calculate trimmed duration (in seconds) of an audio file
    :param file_path: WAV file path
    :return: duration (in seconds)
    """
    y, sr = librosa.load(path=file_path, sr=None)
    yt, _ = librosa.effects.trim(y=y, top_db=20)
    return librosa.get_duration(y=yt, sr=sr)


def get_bidu_asr_params() -> {}:
    """Get Baidu ASR parameters
    :return: parameters
    """
    url = 'https://aip.baidubce.com/oauth/2.0/token'
    params = {'grant_type': 'client_credentials',
              'client_id': 'hnG90vv223EOsdzyPxSrxjbG',
              'client_secret': '2f2toBA4VkFiVGUiVRTngLfcP8hPlW6e'}
    response = requests.get(url, params=params)
    access_token = response.json()['access_token']
    params = {'cuid': hex(uuid.getnode()), 'token': access_token}
    return params


def run_bidu_asr(params: {},
                 file_path: str = None,
                 waveform: np.ndarray = None) -> {}:
    """Run Baidu ASR on either file_path or waveform input
    :param params: Baidu ASR parameters
    :param file_path: WAV file path
    :param waveform: WAV waveform in shape of (len(waveform),)
    :return: ASR result
    """
    url = 'https://vop.baidu.com/server_api'
    wav = load_wav_mono(file_path) if file_path else waveform
    wav = tf.reshape(wav, [len(wav), 1])
    data = tf.audio.encode_wav(wav, SAMPLE_RATE).numpy()
    headers = {'content-type': f'audio/wav;rate={SAMPLE_RATE}'}
    response = requests.post(url, data=data, params=params, headers=headers)
    res = response.json()
    return res


def load_wav_mono(file_path: str) -> tf.Tensor:
    """Load a WAV file, convert it to a float tensor,
    resample to SAMPLE_RATE (16 kHz) single-channel audio
    :param file_path: a tensor of file path
    :return: waveforms with 16 kHz sample rate
    """
    file_contents = tf.io.read_file(file_path)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    if tf.math.not_equal(sample_rate, SAMPLE_RATE):
        wav = tfio.audio.resample(wav, sample_rate, SAMPLE_RATE)
    return wav


def mfccs_from_waveforms(_wav: tf.Tensor, _n_mfcc: int) -> tf.Tensor:
    """Compute MFCCs of waveforms
    :param _wav: the waveform of shape (batch_size, num_samples)
    :param _n_mfcc: the number of mels used for MFCCs
    :return: the MFCCs
    """
    sample_rate = SAMPLE_RATE

    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.signal.stft(
        _wav, frame_length=1024, frame_step=256, fft_length=1024)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms = tf.ensure_shape(
        mel_spectrograms, spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :_n_mfcc]
    return mfccs


def feature_extraction(file_path: str,
                       dims: [int],
                       gate: [int],
                       offset: float = 0.0,
                       duration: float = None) -> ():
    """Return audio features
    :param file_path: audio file path
    :param dims: feature dimensions (e.g. [20, 1, 1, 1])
    :param gate: feature gate (e.g. [1, 1, 1, 1])
    :param offset: start reading after this time (in seconds)
    :param duration: only load up to this much audio (in seconds)
    :return: audio features
    """
    y, sr = librosa.load(file_path, sr=None, offset=offset, duration=duration)

    mfccs = np.mean(lf.mfcc(y=y, sr=sr, n_mfcc=dims[0]).T, 0) if gate[0] else 0
    rmse = np.mean(lf.rms(y=y).T, 0) if gate[1] else 0
    sf = np.mean(lo.onset_strength(y=y, sr=sr).T, 0) if gate[2] else 0
    zcr = np.mean(lf.zero_crossing_rate(y=y).T, 0) if gate[3] else 0
    extracted_features = np.hstack([mfccs, rmse, sf, zcr])
    extracted_features = extracted_features[extracted_features != 0]
    return extracted_features


def parse_audio_files(parent_dir: str,
                      sub_dirs: [str],
                      feature_dims: [int],
                      feature_gate: [int],
                      file_ptn: str = '*') -> ():
    """Audio parsing, return array with features and labels
    :param parent_dir: parent directory where audio files are stored
    :param sub_dirs: subdirectories that are in the parent directory
    :param feature_dims: feature dimensions (e.g. [20, 1, 1, 1])
    :param feature_gate: feature gate (e.g. [1, 1, 1, 1])
    :param file_ptn: audio file name pattern
    :return: array with features, labels and filenames
    """
    number_of_features = np.dot(feature_dims, feature_gate)
    features, labels = np.empty((0, number_of_features)), np.empty(0)
    file_names = []

    for label, sub_dir in enumerate(sub_dirs):
        file_count = 0
        for file_name in glob.glob(os.path.join(parent_dir, sub_dir, file_ptn)):
            try:
                extracted_features = feature_extraction(file_name,
                                                        dims=feature_dims,
                                                        gate=feature_gate)
                file_count += 1
            except Exception as e:
                print(f'Error in feature extraction from {file_name}: {e}')
                continue
            features = np.vstack([features, extracted_features])
            labels = np.append(labels, label)
            file_names.append(file_name)
        print(f'Extracted features from \'{sub_dir}\', {file_count} file(s);')
    return np.array(features), np.array(labels, dtype=np.int), file_names


def visualize_mfcc_series(features: np.ndarray):
    """Visualize the MFCC series
    :param features: audio MFCC features
    """
    plt.figure()
    ld.specshow(features, x_axis='time')
    plt.colorbar()
    plt.title('MFCC for Audio Frame')
    plt.tight_layout()
    plt.show()


def visualize_spectrograms(file_path: str):
    """Visualize spectrograms of an audio file
    :param file_path: audio file path
    """
    y, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(11, 8))
    file_name = os.path.split(file_path)[-1]
    plt.suptitle(file_name, fontsize=16)
    d = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    c = librosa.feature.chroma_cqt(y=y, sr=sr)

    plt.subplot(2, 2, 1)
    ld.specshow(d, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear power spectrogram (grayscale)')

    plt.subplot(2, 2, 2)
    ld.specshow(d, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequence power spectrogram')

    plt.subplot(2, 2, 3)
    ld.specshow(c, y_axis='chroma')
    plt.colorbar()
    plt.title('Chromagram')

    plt.subplot(2, 2, 4)
    ld.specshow(d, y_axis='linear', cmap='gray_r')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear power spectrogram (grayscale)')

    plt.show()
