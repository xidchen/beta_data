import collections


def replace_token_for_bert(_text: str) -> str:
    """Replace token for BERT-zh tokenization and standardization"""
    replace_dict = {'“': '"',
                    '”': '"',
                    '‘': '\'',
                    '’': '\'',
                    '（': '(',
                    '）': ')',
                    '—': '-'}
    for k, v in replace_dict.items():
        _text = _text.replace(k, v)
    return _text.lower()


def split_one_line_long_article(_article: str, _max_len: int) -> [str]:
    """Split a one line long article into paragraphs no longer than
    the maximum length and at sentence terminating punctuations.
    Suppose no sentence is longer than the maximum length."""
    s_terminators = ['。', '！', '？']
    lower_quotation_mark = '”'
    res = []
    if len(_article) <= _max_len:
        res.append(_article)
        return res
    cache_p, cache_s = '', ''
    for _char in _article:
        if _char in s_terminators:
            cache_s += _char
        else:
            if _char == lower_quotation_mark:
                if cache_s[-1] in s_terminators:
                    cache_s += _char
                    if len(cache_p + cache_s) <= _max_len:
                        cache_p += cache_s
                    else:
                        res.append(cache_p)
                        cache_p = cache_s
                    cache_s = ''
                else:
                    cache_s += _char
            else:
                if cache_s and cache_s[-1] in s_terminators:
                    if len(cache_p + cache_s) <= _max_len:
                        cache_p += cache_s
                    else:
                        res.append(cache_p)
                        cache_p = cache_s
                    cache_s = _char
                else:
                    cache_s += _char
    if cache_s:
        if len(cache_p + cache_s) <= _max_len:
            cache_p += cache_s
        else:
            res.append(cache_p)
            cache_p = cache_s
    if cache_p:
        res.append(cache_p)
    return res


def split_paragraph(_paragraph: str,
                    _max_len: int,
                    _split: bool = False) -> [str]:
    """Split paragraph at sentence terminating punctuations"""
    s_terminators = ['。', '！', '？']
    lower_quotation_mark = '”'
    res = []
    if _split:
        if len(_paragraph) <= _max_len:
            res.append(_paragraph)
            return res
        cache = ''
        for _char in _paragraph:
            if _char in s_terminators:
                cache += _char
            else:
                if _char == lower_quotation_mark:
                    if cache[-1] in s_terminators:
                        res.append(cache + _char)
                        cache = ''
                    else:
                        cache += _char
                else:
                    if cache and cache[-1] in s_terminators:
                        res.append(cache)
                        cache = _char
                    else:
                        cache += _char
        if cache:
            res.append(cache)
    else:
        res.append(_paragraph)
    return res


def split_long_sentence(_sentence: str, _max_len: int) -> []:
    """Split sentence at Chinese comma if length over max length,
    and keep an overlap as long as possible to maintain coherence"""
    comma = '，'
    res = []
    _subs = _sentence.split(comma)
    _subs[:-1] = [_sub + comma for _sub in _subs[:-1]]
    _cur_sent = collections.deque()
    _cur_len = 0
    for i in range(len(_subs)):
        _cur_len += len(_subs[i])
        if _cur_len >= _max_len and _cur_sent:
            res.append(''.join(_cur_sent))
        _cur_sent.append(_subs[i])
        while _cur_len > _max_len:
            _cur_len -= len(_cur_sent.popleft())
    if _cur_sent:
        res.append(''.join(_cur_sent))
    return res
