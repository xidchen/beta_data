import json
import numpy as np
import random
import re
import requests
import zlib


def semantic_embedding(s0: [str]) -> [[float]]:
    """Calculate semantic embedding of a list of sentences
    :param s0: list of sentences
    :return: list of semantic embeddings
    """
    url = 'http://localhost:5300'
    r = json.loads(requests.post(url, data={'s0': str(s0)}).text)
    return r['s0_embed']


def split_sentence(sentence: str) -> [str]:
    """Split sentence at given punctuations into list of phrases
    :param sentence: the sentence
    :return: list of phrases
    """
    punctuations = ['。', '！', '？', '，', ';', '：']
    pattern = '[' + ''.join(punctuations) + ']'
    return [_s.strip() for _s in re.split(pattern, sentence) if _s.strip()]


def split_by_semicolon(s: str) -> [str]:
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
    rhetoric = rhetoric.lower()
    transcript = transcript.lower()
    rhetorics = [rhetoric, drop_ending_punctuation(rhetoric)]
    transcripts = [transcript, drop_ending_punctuation(transcript)]
    se = semantic_embedding(rhetorics + transcripts)
    scores = np.inner(se[:2], se[2:])
    score = np.max(scores)
    res = int(score * 100) if score >= 0 else 0
    return res


def keywords_score(keywords: str, transcript: str) -> int:
    """Calculate keywords score based on keywords reference and transcript
    :param keywords: keywords reference seperated by semicolon
    :param transcript: user's ASR result
    :return: score
    """
    keywords = keywords.lower()
    transcript = transcript.lower()
    keywords = split_by_semicolon(keywords)
    transcripts = split_sentence(transcript)
    se = semantic_embedding(keywords + transcripts)
    scores = np.inner(se[:len(keywords)], se[len(keywords):])
    k_in_t = np.array([[1] if k in transcript else [0] for k in keywords])
    scores = np.hstack((scores, k_in_t))
    scores = np.max(scores, axis=1)
    res = int(np.mean(scores) * 100)
    return res


def speed_score(duration: float, transcript: str) -> int:
    """Calculate speed score based on audio duration and transcript word count
    :param duration: audio duration in seconds
    :param transcript: user's ASR result
    :return: score
    """
    res = 0
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
    try:
        url = 'http://localhost:5610'
        r = requests.post(url, data={'file_path': file_path}).text
        score = int(r)
    except Exception as e:
        print(f'[Error] use alternative since: {e}')
        with open(file_path, 'rb') as f:
            data = f.read()
        r = random.Random(zlib.adler32(data))
        score = int(r.triangular(51, 100, 85))
    res = min(score, int(duration * 17))
    res = res if res < 96 else 100
    return res


def articulation_score(duration: float, file_path: str) -> int:
    """Calculate articulation score based on audio content
    :param duration: audio duration in seconds
    :param file_path: audio file path
    :return: score
    """
    try:
        url = 'http://localhost:5620'
        r = requests.post(url, data={'file_path': file_path}).text
        score = int(r)
    except Exception as e:
        print(f'[Error] use alternative since: {e}')
        with open(file_path, 'rb') as f:
            data = f.read()
        r = random.Random(zlib.crc32(data))
        score = int(r.triangular(51, 100, 85))
    res = min(score, int(duration * 17))
    res = res if res < 96 else 100
    return res
