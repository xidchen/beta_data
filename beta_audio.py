import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import tensorflow as tf
import tensorflow_io as tfio
import uuid

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


SAMPLE_RATE = 16000
ALLOWED_EXTENSIONS = {'wav'}


def allowed_file(name: str) -> bool:
    """Check whether the file name has an allowed extension
    :param name: audio file name
    :return: True or False
    """
    return '.' in name and name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def replace_ext_to_txt(name: str) -> str:
    """Repalce the file extension to .txt
    :param name: original file name
    :return: file name whose extension replaced with .txt
    """
    return name.replace(name.rsplit('.', 1)[1], 'txt')


def run_bidu_asr(file_path: str) -> {}:
    """Run Baidu asr
    :param file_path: WAV file path
    :return: asr result
    """
    url = 'https://aip.baidubce.com/oauth/2.0/token'
    params = {'grant_type': 'client_credentials',
              'client_id': 'hnG90vv223EOsdzyPxSrxjbG',
              'client_secret': '2f2toBA4VkFiVGUiVRTngLfcP8hPlW6e'}
    response = requests.get(url, params=params)
    access_token = response.json()['access_token']
    params = {'cuid': hex(uuid.getnode()), 'token': access_token}

    url = 'https://vop.baidu.com/server_api'
    wav = load_wav_mono(file_path)
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


def feature_extraction(file_path: str) -> ():
    """Return audio features
    :param file_path: audio file path
    :return: audio features
    """
    y, sr = librosa.load(file_path, sr=None)
    if y.ndim > 1:
        y = y[:, 0]
    y = y.T

    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
    rmse = np.mean(librosa.feature.rms(y=y).T, axis=0)
    spectral_flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    return mfccs, rmse, spectral_flux, zcr


def parse_audio_files(parent_dir: str,
                      sub_dirs: list,
                      file_ext: str = '*.wav') -> ():
    """Audio parsing, return array with features and labels
    :param parent_dir: parent directory where audio files are stored
    :param sub_dirs: subdirectories that are in the parent directory
    :param file_ext: audio file extension
    :return: array with features and labels
    """
    n_mfccs = 20
    number_of_features = n_mfccs + 3
    features, labels = np.empty((0, number_of_features)), np.empty(0)

    for label, sub_dir in enumerate(sub_dirs):
        for file_name in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            print(f'Actual file name: {file_name}')
            try:
                mfccs, rmse, spectral_flux, zcr = feature_extraction(file_name)
            except Exception as e:
                print(f'[Error] There was an error in feature extraction: {e}')
                continue
            extracted_features = np.hstack([mfccs, rmse, spectral_flux, zcr])
            features = np.vstack([features, extracted_features])
            labels = np.append(labels, label)
        print(f'Extracted features from {sub_dir}, done')
    return np.array(features), np.array(labels, dtype=np.int)


def visualize_mfcc_series(features: np.ndarray):
    """Visualize the MFCC series
    :param features: audio MFCC features
    """
    plt.figure()
    librosa.display.specshow(features, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs = 5 Values for 5s Audio Frames (High Class)')
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
    librosa.display.specshow(d, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear power spectrogram (grayscale)')

    plt.subplot(2, 2, 2)
    librosa.display.specshow(d, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequence power spectrogram')

    plt.subplot(2, 2, 3)
    librosa.display.specshow(c, y_axis='chroma')
    plt.colorbar()
    plt.title('Chromagram')

    plt.subplot(2, 2, 4)
    librosa.display.specshow(d, y_axis='linear', cmap='gray_r')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear power spectrogram (grayscale)')

    plt.show()
