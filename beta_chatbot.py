import json
import requests

import beta_code


def run_bert_intent(query: str) -> str:
    """Run BERT intent model on query
    :param query: the query
    :return: intent of query by BERT intent model
    """
    u = 'http://localhost:5410'
    return json.loads(requests.post(u, data={'query': query}).text)['intent']


def run_bert_entity(query: str) -> [[int, int, str]]:
    """Run BERT entity model on query
    :param query: the query
    :return: list of start, end positions and type of entities
    """
    u = 'http://localhost:5420'
    return json.loads(requests.post(u, data={'query': query}).text)['entities']


def get_query_entity(query: str,
                     verbose: bool,
                     entities: [[int, int, str]],
                     e_name_to_code: {},
                     e_code_to_name: {}) -> [{}]:
    """Get entity code, text and type from positions of entities in the query
    :param query: the query
    :param verbose: whether to print extension information
    :param entities: list of start, end positions and type of entities
    :param e_name_to_code: dict of entity name to code
    :param e_code_to_name: dict of entity code to name
    :return: list of code, text and type of entities
    """
    res = []
    for entity in entities:
        e_text = query[entity[0]:entity[1]]
        e_type = entity[2]
        e_code = e_name_to_code.get(e_type, {}).get(e_text, '')
        if isinstance(e_code, str):
            if not e_code:
                try:
                    guess = beta_code.get_guess_code(e_text, e_type)
                    if guess:
                        for e_code in guess:
                            if verbose:
                                e_name = e_code_to_name.get(
                                    e_type, {}).get(e_code, '')
                                res.append(
                                    {'code': e_code, 'text': e_text,
                                     'guess': guess[e_code].lower(),
                                     'type': e_type, 'name': e_name})
                            else:
                                res.append(
                                    {'code': e_code, 'text': e_text,
                                     'guess': guess[e_code].lower(),
                                     'type': e_type})
                        continue
                except Exception:
                    pass
            if verbose:
                e_name = e_code_to_name.get(e_type, {}).get(e_code, '')
                res.append(
                    {'code': e_code, 'text': e_text, 'type': e_type,
                     'name': e_name})
            else:
                res.append(
                    {'code': e_code, 'text': e_text, 'type': e_type})
        if isinstance(e_code, list):
            e_code_list = e_code
            for e_code in e_code_list:
                if verbose:
                    e_name = e_code_to_name.get(e_type, {}).get(e_code, '')
                    res.append(
                        {'code': e_code, 'text': e_text, 'type': e_type,
                         'name': e_name})
                else:
                    res.append(
                        {'code': e_code, 'text': e_text, 'type': e_type})
    return res
