import re


def replace_token_for_bert(_text: str) -> str:
    replace_dict = {'“': '"',
                    '”': '"',
                    '‘': '\'',
                    '’': '\'',
                    '—': '-'}
    for k, v in replace_dict.items():
        _text = _text.replace(k, v)
    return _text.lower()


def replace_whitespace_in_pattern(_text: str) -> str:
    if re.fullmatch(r'(\d{6}\s+)+\d{6}', _text):
        return re.sub(r'\s+', '，', _text)
    return _text
