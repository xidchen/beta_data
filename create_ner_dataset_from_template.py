import json
import os
import random
import re

import beta_code


entity_class_path = './entity_classes.txt'
with open(entity_class_path, encoding='utf-8') as f:
    entity_classes = f.read().strip().split('\n')
print(f'Entity class {entity_class_path} loaded')
entity_pool = {}
for e_class in entity_classes:
    try:
        entities = list(beta_code.get_entity_code(e_class).keys())
        print(f'Entity list {e_class} loaded, {len(entities)} entities')
        entity_pool[e_class] = entities
    except KeyError:
        print(f'Entity list {e_class} not loaded')
ner_template_path = './ner_template.txt'
with open(ner_template_path, encoding='utf-8') as f:
    ner_templates = f.read().strip().split('\n')

root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_dir = os.path.join(root_dir, 'BetaData')
data_file_str = os.path.join(data_root_dir, 'demo_beta_ner_from_template.jsonl')

e_class_isdigit = '基金产品'
placeholder_isdigit = '[' + e_class_isdigit + ']'
t_isdigit = [
    j.join([placeholder_isdigit] * i) for i in range(2, 6) for j in [' ', ', ']]
e_isdigit = [e for e in entity_pool[e_class_isdigit] if e.isdigit()]

ds = []
copy_size = 1000
start_id = 10000
for i, template in enumerate(ner_templates):
    for j in range(copy_size):
        t_copy = template
        t_accept = True
        t_labels = []
        e_in_t = []
        if t_copy in t_isdigit:
            e_class = e_class_isdigit
            placeholder = placeholder_isdigit
            e_in_c = []
            if placeholder in t_copy and e_class in entity_pool:
                for _ in range(t_copy.count(placeholder)):
                    entity = random.choice(e_isdigit)
                    while entity in e_in_c:
                        entity = random.choice(e_isdigit)
                    t_copy = t_copy.replace(placeholder_isdigit, entity, 1)
                    e_in_c.append(entity)
                    e_in_t.append((entity, e_class))
            if placeholder in t_copy and e_class not in entity_pool:
                t_accept = False
                break
        else:
            for e_class in entity_classes:
                placeholder = '[' + e_class + ']'
                e_in_c = []
                if placeholder in t_copy and e_class in entity_pool:
                    for _ in range(t_copy.count(placeholder)):
                        entity = random.choice(entity_pool[e_class])
                        while entity in e_in_c:
                            entity = random.choice(entity_pool[e_class])
                        t_copy = t_copy.replace(placeholder, entity, 1)
                        e_in_c.append(entity)
                        e_in_t.append((entity, e_class))
                if placeholder in t_copy and e_class not in entity_pool:
                    t_accept = False
                    break
        if t_accept:
            e_in_t_more_than_once = []
            for (e, e_class) in e_in_t:
                if t_copy.count(e) == 1:
                    t_labels.append(
                        [t_copy.index(e), t_copy.index(e) + len(e), e_class])
                if t_copy.count(e) > 1:
                    e_in_t_more_than_once.append((e, e_class))
            t_labels.sort()
            for (e, e_class) in e_in_t_more_than_once:
                for e_span in re.finditer(e, t_copy):
                    e_span_accept = True
                    for t_span in t_labels:
                        if (t_span[0] <= e_span.start() < t_span[1] or
                                t_span[0] < e_span.end() <= t_span[1]):
                            e_span_accept = False
                            break
                    if e_span_accept:
                        t_labels.append([e_span.start(), e_span.end(), e_class])
                        break
            t_labels.sort()
            ds.append(
                {'id': start_id + i * copy_size + j, 'text': t_copy, 'meta': {},
                 'annotation approver': None, 'labels': t_labels})

with open(data_file_str, mode='w', encoding='utf-8') as f:
    for d in ds:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')
print(f'{data_file_str} done')
exit(0)
