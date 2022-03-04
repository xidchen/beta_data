import librosa
import numpy as np
import re
import requests


def semantic_similarity(s1: str, s2: str) -> int:
    """Calculate semantic similarity of two sentences
    :param s1: first sentence
    :param s2: second sentence
    :return: semantic similarity
    """
    url = 'http://localhost:5000/semantic_similarity'
    res = 0
    if s1 and s2:
        res = requests.post(url, data={'s1': s1, 's2': s2}).text
        res = float(res)
    return res


def split_sentence(sentence: str) -> []:
    """Split sentence at given punctuations into list of phrases
    :param sentence: the sentence
    :return: list of phrases
    """
    punctuations = ['。', '！', '？', '，', ';', '：']
    pattern = '[' + ''.join(punctuations) + ']'
    res = re.split(pattern, sentence)
    return res


def rhetoric_score(rhetoric: str, transcript: str) -> int:
    """Calculate rhetoric score based on rhetoric reference and transcript
    :param rhetoric: rhetoric reference
    :param transcript: user's ASR result
    :return: score
    """
    score = semantic_similarity(rhetoric, transcript)
    res = int(score * 100) if score >= 0 else 0
    return res


def keywords_score(keywords: str, transcript: str) -> int:
    """Calculate keywords score based on keywords reference and transcript
    :param keywords: keywords reference seperated by semicolon
    :param transcript: user's ASR result
    :return: score
    """
    res = 0
    if keywords and transcript:
        keywords = keywords.split(sep=';')
        transcripts = split_sentence(transcript)
        scores = []
        for k in keywords:
            _scores = []
            for t in transcripts:
                _score = 1 if k in t else semantic_similarity(k, t)
                _scores.append(_score)
            scores.append(np.max(_scores))
        res = int(np.mean(scores) * 100)
    return res


def speed_score(audio_file_path: str, transcript: str) -> int:
    """Calculate speed score based on audio duration and transcript word count
    :param audio_file_path: audio file path in the file system
    :param transcript: user's ASR result
    :return: score
    """
    res = 0
    s_per_m = 60
    audio_duration = librosa.get_duration(filename=audio_file_path)
    word_count = len(transcript)
    if audio_duration and word_count:
        speed = int(word_count / audio_duration * s_per_m)
        if 60 <= speed < 120 or 300 <= speed < 350:
            res = 60
        if 120 <= speed < 300:
            res = 100
    return res
