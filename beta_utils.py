def replace_token_for_bert(_text: str) -> str:
    """
    Replace token for BERT-zh tokenization and standardization
    """
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
