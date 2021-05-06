import json
import requests


def get_intent_code() -> ({}, {}):
    """Fetch Intent API, get mapping of intent name and id"""
    intent_url = 'https://ws7.betawm.com/betacorpus/api/Purposes/all'
    response = requests.get(intent_url)
    content = response.text
    intent_name_to_id, intent_id_to_name = dict(), dict()
    if content:
        rs = json.loads(content)
        for v in rs:
            intent_name_to_id[str(v['name'])] = str(v['id'])
            intent_id_to_name[str(v['id'])] = str(v['name'])
    return intent_name_to_id, intent_id_to_name


def get_entity_code(entity_class: str) -> {}:
    """Fetch Entity API, get mapping of entity name and code for each class"""
    urls = dict()
    urls['基金产品'] = 'https://ws7.betawm.com/betacorpus/api/SlotEntity/get/1'
    urls['基金经理'] = 'https://ws7.betawm.com/betacorpus/api/SlotEntity/get/2'
    urls['基金公司'] = 'https://ws7.betawm.com/betacorpus/api/SlotEntity/get/3'
    urls['基金主题行业'] = 'https://ws7.betawm.com/betacorpus/api/SlotEntity/get/5'
    response = requests.get(urls[entity_class])
    content = response.text
    entity_name_to_code = dict()
    if content:
        rs = json.loads(content)
        e_ids = rs['data']['Id']
        e_names = rs['data']['Name']
        for i in range(len(e_ids)):
            entity_name_to_code[str(e_names[i])] = str(e_ids[i])
    return entity_name_to_code
