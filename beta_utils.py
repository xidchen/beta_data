import collections
import pandas as pd
import re


def replace_token_for_bert(text: str) -> str:
    """Replace token for BERT-zh tokenization and standardization
    :param text: the input text
    :return: the output text with token replaced
    """
    replace_dict = {'“': '"',
                    '”': '"',
                    '‘': '\'',
                    '’': '\'',
                    '（': '(',
                    '）': ')',
                    '—': '-'}
    for k, v in replace_dict.items():
        text = text.replace(k, v)
    return text.lower()


def split_one_line_long_article(article: str, max_len: int) -> [str]:
    """Split a one line long article into paragraphs as long as possible but
    no longer than the maximum length, and at sentence terminating punctuation;
    As long as the sentence is shorter than the maximum length,
    sentence terminating punctuation can be skipped
    :param article: the article in one line
    :param max_len: the maximum length after splitting
    :return: list of sentences after splitting
    """
    s_terminators = ['。', '！', '？']
    lower_quotation_mark = '”'
    res = []
    if len(article) <= max_len:
        res.append(article)
        return res
    cache_p, cache_s = '', ''
    for char in article:
        if char in s_terminators:
            cache_s += char
        else:
            if char == lower_quotation_mark:
                if cache_s and cache_s[-1] in s_terminators:
                    cache_s += char
                    if len(cache_p + cache_s) <= max_len:
                        cache_p += cache_s
                    else:
                        res.append(cache_p)
                        cache_p = cache_s
                    cache_s = ''
                else:
                    cache_s += char
            else:
                if cache_s and cache_s[-1] in s_terminators:
                    if len(cache_p + cache_s) <= max_len:
                        cache_p += cache_s
                    else:
                        if cache_p:
                            res.append(cache_p)
                        cache_p = cache_s
                    cache_s = char
                else:
                    cache_s += char
    if cache_s:
        if len(cache_p + cache_s) <= max_len:
            cache_p += cache_s
        else:
            if cache_p:
                res.append(cache_p)
            cache_p = cache_s
    if cache_p:
        res.append(cache_p)
    return res


def split_paragraph(paragraph: str,
                    max_len: int,
                    split: bool = False) -> [str]:
    """Split paragraph at sentence terminating punctuations
    :param paragraph: the input paragraph
    :param max_len: the maximum length
    :param split: whether to split
    :return: list of sentences
    """
    s_terminators = ['。', '！', '？']
    lower_quotation_mark = '”'
    res = []
    if split:
        if len(paragraph) <= max_len:
            res.append(paragraph)
            return res
        cache = ''
        for char in paragraph:
            if char in s_terminators:
                cache += char
            else:
                if char == lower_quotation_mark:
                    if cache[-1] in s_terminators:
                        res.append(cache + char)
                        cache = ''
                    else:
                        cache += char
                else:
                    if cache and cache[-1] in s_terminators:
                        res.append(cache)
                        cache = char
                    else:
                        cache += char
        if cache:
            res.append(cache)
    else:
        res.append(paragraph)
    return res


def split_long_sentence(sentence: str, max_len: int) -> [str]:
    """Split sentence at Chinese comma if length is over max length,
    and keep the overlap as long as possible to maintain coherence
    :param sentence: the long sentence
    :param max_len: the maximum length after splitting
    :return: list of sentences
    """
    comma = '，'
    res = []
    subs = sentence.split(comma)
    subs[:-1] = [sub + comma for sub in subs[:-1]]
    cur_sent = collections.deque()
    cur_len = 0
    for i in range(len(subs)):
        cur_len += len(subs[i])
        if cur_len >= max_len and cur_sent:
            res.append(''.join(cur_sent))
        cur_sent.append(subs[i])
        while cur_len > max_len:
            cur_len -= len(cur_sent.popleft())
    if cur_sent:
        res.append(''.join(cur_sent))
    return res


def adjust_label_offset(labels: [[int, int, str]],
                        span: [int, int]) -> [[int, int, str]]:
    """Adjust label offsets according to the sub-sentence span in the original;
    concretely, select label and adjust its offsets if located in the span
    :param labels: the start, end and type of entities in the original sentence
    :param span: the span of sub-sentence in the original sentence
    :return: the start, end and type of entities in the sub-sentence
    """
    res = []
    for label in labels:
        if span[0] <= label[0] and label[1] <= span[1]:
            res.append([label[0] - span[0], label[1] - span[0], label[2]])
    return res


def sort_numerically(los: [str]) -> [str]:
    """Sort a list numerically
    :param los: list of strings
    :return: list of strings
    """
    return sorted(los, key=alphanum_key)


def alphanum_key(s: str) -> [int or str]:
    """Turn a string into a list of string and number chunks
    :param s: a string
    :return: list of strings or integers
    """
    return [try_int(c) for c in re.split(r'(\d+)', s)]


def try_int(s: str) -> int or str:
    """Return an int if possible, or str unchanged
    :param s: a string
    :return: convert str to int if the string is a number otherwise unchanged
    """
    return int(s) if s.isdigit() else s


def is_cjk_character(char: str) -> bool:
    """Check whether a character is a CJK character
    :param char: the character
    :return: whether the character is a CJK character
    """
    return True if int(0x4e00) <= ord(char) <= int(0x9fff) else False


def drop_unnecessary_whitespace(sentence: str) -> str:
    """Remove unnecessary whitespace in a sentence
    :param sentence: the sentence
    :return: the sentence without unnecessary whitespace
    """
    whitespace = ' '
    double_whitespace = whitespace * 2
    while double_whitespace in sentence:
        sentence = sentence.replace(double_whitespace, whitespace)
    sentence = sentence.strip()
    whitespace_counts = sentence.count(whitespace)
    start = 0
    for _ in range(whitespace_counts):
        w_index = start + sentence[start:].find(whitespace)
        char_l, char_r = sentence[w_index - 1], sentence[w_index + 1]
        if is_cjk_character(char_l) or is_cjk_character(char_r):
            sentence = sentence[:w_index] + sentence[w_index + 1:]
        start = w_index + 1
    return sentence


def drop_unnecessary_whitespace_in_series(series: pd.Series) -> pd.Series:
    """Remove unnecessary whitespace in a pandas Series
    :param series: the pandas Series
    :return: the pandas Series without unnecssary whitespace
    """
    return series.apply(drop_unnecessary_whitespace)
