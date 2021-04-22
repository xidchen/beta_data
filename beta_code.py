import json
import requests


def get_intent_code():
    intent_url = 'http://172.17.12.174:31080/betacorpus/api/Purposes/all'
    response = requests.get(intent_url)
    content = response.text
    intent_id_name = dict()
    if content:
        rs = json.loads(content)
        for v in rs:
            intent_id_name[v['name']] = v['id']
    return intent_id_name


def get_entity_code(entity_class):
    urls = dict()
    urls['基金产品'] = 'https://ws7.betawm.com/betacorpus/api/SlotEntity/get/1'
    urls['基金经理'] = 'https://ws7.betawm.com/betacorpus/api/SlotEntity/get/2'
    urls['基金公司'] = 'https://ws7.betawm.com/betacorpus/api/SlotEntity/get/3'
    urls['基金主题行业'] = 'https://ws7.betawm.com/betacorpus/api/SlotEntity/get/5'
    response = requests.get(urls[entity_class])
    content = response.text
    entity_to_id = dict()
    if content:
        rs = json.loads(content)
        ids = rs['data']['Id']
        entity = rs['data']['Name']
        for i in range(len(ids)):
            entity_to_id[entity[i]] = ids[i]
    return entity_to_id
