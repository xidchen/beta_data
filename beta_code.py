import json
import requests

import beta_utils


def get_intent_code() -> ({}, {}):
    """Fetch intent API, get mapping of intent name and id"""
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


def get_entity_code(entity_class: str) -> ({}, {}):
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
    entity_name_to_code, entity_code_to_name = dict(), dict()
    if content:
        rs = json.loads(content)
        e_ids = rs['data']['Id']
        e_names = rs['data']['Name']
        for i in range(len(e_ids)):
            e_name = beta_utils.replace_token_for_bert(str(e_names[i]).strip())
            if e_name in entity_name_to_code:
                if isinstance(entity_name_to_code[e_name], str):
                    entity_name_to_code[e_name] = [entity_name_to_code[e_name]]
                if isinstance(entity_name_to_code[e_name], list):
                    entity_name_to_code[e_name].append(str(e_ids[i]))
            else:
                entity_name_to_code[e_name] = str(e_ids[i])
            e_id = e_ids[i]
            if e_id in entity_code_to_name:
                if isinstance(entity_code_to_name[e_id], str):
                    entity_code_to_name[e_id] = [entity_code_to_name[e_id]]
                if isinstance(entity_code_to_name[e_id], list):
                    entity_code_to_name[e_id].append(str(e_names[i]).strip())
            else:
                entity_code_to_name[e_id] = str(e_names[i]).strip()
    return entity_name_to_code, entity_code_to_name


def get_perspective_and_event(event_classes: []) -> ([{}, {}], {}):
    """Fetch wp API, get mapping of wp name and id,
    and mapping of wp and event"""
    wp_url = 'https://as2.betawm.com/Beta.BigDataTagAPI/api/event/tree'
    response = requests.get(wp_url)
    content = response.text
    wp_name_to_id, wp_id_to_name, wp_to_event = {}, {}, {}
    if content:
        rs = json.loads(content)
        for i, n1 in enumerate(rs['Data']['Nodes']):
            if i in event_classes:
                for n2 in n1['Nodes']:
                    for n3 in n2['Nodes']:
                        for n4 in n3['Nodes']:
                            wp_name_to_id[str(n4['Name'])] = str(n4['Id'])
                            wp_id_to_name[str(n4['Id'])] = str(n4['Name'])
                            wp_to_event[str(n4['Name'])] = str(n3['Name'])
    return [wp_name_to_id, wp_id_to_name], wp_to_event
