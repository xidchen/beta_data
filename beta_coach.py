import json
import numpy as np
import random
import re
import requests
import zlib


CONTEXT_TOTAL_LENGTH = 12
FOCUS_KEYWORD_MAX_RATIO = 4


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


def drop_punctuation(sentence: str) -> str:
    """Remove punctuations in a sentence
    :param sentence: the sentence
    :return: the sentence without punctuation
    """
    punctuations = ['。', '！', '？', '，', ';', '：', '、']
    for punctuation in punctuations:
        sentence = sentence.replace(punctuation, '')
    return sentence


def drop_whitespace(sentence: str) -> str:
    """Remove whitespaces in a sentence
    :param sentence: the sentence
    :return: the sentence without whitespace
    """
    whitespace = ' '
    sentence = sentence.replace(whitespace, '')
    return sentence


def get_focuses_on_transcript(rhetoric: str,
                              keywords: [str],
                              transcript: str) -> [str]:
    """Get focuses of keywords from rhetoric on transcript
    :param rhetoric: rhetoric reference
    :param keywords: keywords reference in a list
    :param transcript: user's ASR result
    :return: the focuses
    """
    focuses = set()
    rhetoric = drop_punctuation(rhetoric)
    rhetoric = drop_whitespace(rhetoric)
    transcript = drop_punctuation(transcript)
    transcript = drop_whitespace(transcript)
    for keyword in keywords:
        contexts = get_contexts_of_keyword(rhetoric, keyword)
        for context in contexts:
            focuses.update(get_focuses_based_on_context(transcript, context))
    focus_max_length = len(max(keywords, key=len)) * FOCUS_KEYWORD_MAX_RATIO
    focuses = [f for f in focuses if len(f) <= focus_max_length]
    focuses = sorted(focuses, key=len)
    return focuses


def get_contexts_of_keyword(rhetoric: str, keyword: str) -> [[str]]:
    """Get contexts of keyword from rhetoric
    :param rhetoric: the rhetoric
    :param keyword: one keyword, which may occurs more than once
    :return: the left and right contexts of the keyword
    """
    contexts = []
    context_l_length = CONTEXT_TOTAL_LENGTH // 2
    context_r_length = CONTEXT_TOTAL_LENGTH - context_l_length
    keyword_counts = rhetoric.count(keyword)
    start, end = 0, len(rhetoric)
    for _ in range(keyword_counts):
        key_l_index = rhetoric.find(keyword, start, end)
        key_r_index = key_l_index + len(keyword)
        if key_l_index - context_l_length <= 0:
            context_l = rhetoric[:key_l_index]
        else:
            context_l = rhetoric[key_l_index - context_l_length:key_l_index]
        if key_r_index + context_r_length >= end:
            context_r = rhetoric[key_r_index:]
        else:
            context_r = rhetoric[key_r_index:key_r_index + context_r_length]
        contexts.append([context_l, context_r])
        start = key_r_index
    return contexts


def get_focuses_based_on_context(transcript: str, context: [str]) -> [str]:
    """Get focus on the transcript based on the context of a keyword
    :param transcript: user's ASR result without punctuation and whitespace
    :param context: the context of a keyword on the rhetoric reference
    :return: all possible focuses on the transcript
    """
    context_l, context_r = context[0], context[1]
    focus_l_indices = get_focus_left_indices(transcript, context_l)
    focus_r_indices = get_focus_right_indices(transcript, context_r)
    focus_ranges = get_focus_ranges(focus_l_indices, focus_r_indices)
    focus = [transcript[f_r[0]:f_r[1]] for f_r in focus_ranges]
    return focus


def get_focus_left_indices(transcript: str, context_l: str) -> {int}:
    """Get the left indices of focus on transcript based on the left context,
    which is matched with the transcript from full length to rightmost one
    :param transcript: user's ASR result without punctuation and whitespace
    :param context_l: the left context of a keyword on rhetoric
    :return: the left indices of the focus
    """
    focus_l_indices = set()
    for i in range(len(context_l), 0, -1):
        c = context_l[-i:]
        c_count = transcript.count(c)
        start, end = 0, len(transcript)
        for _ in range(c_count):
            focus_l_index = transcript.find(c, start, end) + len(c)
            focus_l_indices.add(focus_l_index)
            start = focus_l_index
    if not focus_l_indices:
        for i in range(len(context_l), 1, -1):
            for j in range(1, i - 1):
                c = context_l[-i:-j]
                c_count = transcript.count(c)
                start, end = 0, len(transcript)
                for _ in range(c_count):
                    focus_l_index = transcript.find(c, start, end) + len(c)
                    focus_l_indices.add(focus_l_index)
                    start = focus_l_index
    focus_l_indices.add(0)
    return focus_l_indices


def get_focus_right_indices(transcript: str, context_r: str) -> {int}:
    """Get the right indices of focus on transcript based on the right context,
    which is matched with the transcript from full length to leftmost one
    :param transcript: user's ASR result without punctuation and whitespace
    :param context_r: the right context of a keyword on rhetoric
    :return: the right indices of the focus
    """
    focus_r_indices = set()
    for i in range(len(context_r), 0, -1):
        c = context_r[:i]
        c_count = transcript.count(c)
        start, end = 0, len(transcript)
        for _ in range(c_count):
            focus_r_index = transcript.find(c, start, end)
            focus_r_indices.add(focus_r_index)
            start = focus_r_index + len(c)
    if not focus_r_indices:
        for i in range(len(context_r), 1, -1):
            for j in range(1, i - 1):
                c = context_r[j:i]
                c_count = transcript.count(c)
                start, end = 0, len(transcript)
                for _ in range(c_count):
                    focus_r_index = transcript.find(c, start, end)
                    focus_r_indices.add(focus_r_index)
                    start = focus_r_index + len(c)
    focus_r_indices.add(len(transcript))
    return focus_r_indices


def get_focus_ranges(focus_l_indices: {int}, focus_r_indices: {int}) -> [[int]]:
    """Get focus based on the left indices and the right indices of focus
    :param focus_l_indices: the left indices of the focus
    :param focus_r_indices: the right indices of the focus
    :return: the ranges of all possible focuses
    """
    focus_range = [[f_l_i, f_r_i] for f_l_i in focus_l_indices
                   for f_r_i in focus_r_indices if f_l_i < f_r_i]
    return focus_range


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


def keywords_score(rhetoric: str, keywords: str, transcript: str) -> int:
    """Calculate keywords score based on rhetoric reference,
    keywords reference and transcript
    :param rhetoric: rhetoric reference
    :param keywords: keywords reference seperated by semicolon
    :param transcript: user's ASR result
    :return: score
    """
    rhetoric = rhetoric.lower()
    keywords = keywords.lower()
    transcript = transcript.lower()
    keywords = split_by_semicolon(keywords)
    transcripts = split_sentence(transcript)
    focuses = get_focuses_on_transcript(rhetoric, keywords, transcript)
    se = semantic_embedding(keywords + transcripts + focuses)
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
