import json
import requests


def get_intent_code() -> ({}, {}):
    """Fetch intent API, get mapping of intent name and id"""
    intent_url = 'https://ws7.betawm.com/betacorpus/api/Purposes/all'
    intent_url = 'http://172.17.13.29:31080/betacorpus/api/Purposes/all'
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
    """Fetch entity API, get mapping of entity name and code for each class"""
    urls = dict()
    urls['基金产品'] = 'https://ws7.betawm.com/betacorpus/api/SlotEntity/get/1'
    urls['基金经理'] = 'https://ws7.betawm.com/betacorpus/api/SlotEntity/get/2'
    urls['基金公司'] = 'https://ws7.betawm.com/betacorpus/api/SlotEntity/get/3'
    urls['保险产品'] = 'https://ws7.betawm.com/betacorpus/api/SlotEntity/get/4'
    urls['基金主题行业'] = 'https://ws7.betawm.com/betacorpus/api/SlotEntity/get/5'
    urls['保险险种'] = 'https://ws7.betawm.com/betacorpus/api/SlotEntity/get/6'
    urls['保险公司'] = 'https://ws7.betawm.com/betacorpus/api/SlotEntity/get/7'
    response = requests.get(urls[entity_class])
    content = response.text
    entity_name_to_code = dict()
    if content:
        rs = json.loads(content)
        e_ids = rs['data']['Id']
        e_names = rs['data']['Name']
        for i in range(len(e_ids)):
            e_name = str(e_names[i])
            if e_name in entity_name_to_code:
                if isinstance(entity_name_to_code[e_name], str):
                    entity_name_to_code[e_name] = [entity_name_to_code[e_name]]
                if isinstance(entity_name_to_code[e_name], list):
                    entity_name_to_code[e_name].append(str(e_ids[i]))
            else:
                entity_name_to_code[e_name] = str(e_ids[i])
    return entity_name_to_code