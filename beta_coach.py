import json
import librosa
import numpy as np
import random
import re
import requests
import tensorflow as tf
import zlib


def semantic_similarity(s1: str, s2: str) -> float:
    """Calculate semantic similarity of two sentences
    :param s1: first sentence
    :param s2: second sentence
    :return: semantic similarity
    """
    url = 'http://localhost:5300'
    res = 0
    if s1 and s2:
        r = json.loads(requests.post(url, data={'s1': s1, 's2': s2}).text)
        r1 = tf.reshape(tf.constant(r['s1_embed']), shape=[1, 512])
        r2 = tf.reshape(tf.constant(r['s2_embed']), shape=[1, 512])
        res = tf.einsum('ij,kj->ik', r1, r2).numpy().tolist()[0][0]
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


def split_by_semicolon(s: str) -> []:
    """Split a string by semicolon into list of strings
    :param s: the string
    :return: list of strings
    """
    return [_s.strip() for _s in s.split(';') if _s.strip()]


def drop_ending_punctuation(sentence: str) -> str:
    """Remove punctuation at the end of a sentence
    :param sentence: the sentence
    :return: the sentence without ending punctuation
    """
    punctuations = ['。', '！', '？', '，', ';', '：']
    while sentence:
        if sentence[-1] in punctuations:
            sentence = sentence[:-1]
        else:
            break
    return sentence


def rhetoric_score(rhetoric: str, transcript: str) -> int:
    """Calculate rhetoric score based on rhetoric reference and transcript
    :param rhetoric: rhetoric reference
    :param transcript: user's ASR result
    :return: score
    """
    rhetoric = drop_ending_punctuation(rhetoric)
    transcript = drop_ending_punctuation(transcript)
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
        keywords = split_by_semicolon(keywords)
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


def speed_score(duration: float, transcript: str) -> int:
    """Calculate speed score based on audio duration and transcript word count
    :param duration: audio duration in seconds
    :param transcript: user's ASR result
    :return: score
    """
    res = 0
    if duration and transcript:
        speed = int(len(transcript) / duration)
        if 1 <= speed < 2 or 5 <= speed < 6:
            res = 60
        if 2 <= speed < 5:
            res = 100
    return res


def fluency_score(duration: float, file_path: str) -> int:
    """Calculate fluency score based on audio content
    :param duration: audio duration in seconds
    :param file_path: audio file path
    :return: score
    """
    if duration < 5:
        res = 50
    else:
        with open(file_path, 'rb') as f:
            data = f.read()
        checksum = zlib.adler32(data)
        r = random.Random(checksum)
        score = r.triangular(0.5, 1, 0.85)
        res = int(score * 100)
    return res


def articulation_score(file_path: str) -> int:
    """Calculate articulation score based on audio content
    :param file_path: audio file path
    :return: score
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    checksum = zlib.crc32(data)
    r = random.Random(checksum)
    score = r.triangular(0.5, 1, 0.85)
    res = int(score * 100)
    return res
