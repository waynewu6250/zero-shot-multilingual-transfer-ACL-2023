import glob
import json
import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np


def get_value_dst(DST):
    active_dst = defaultdict(list)
    for k,v in DST.items():
        for k_s, v_s in v['semi'].items():
            if(len(v_s)!=0):
                active_dst[k].append([k_s, v_s])
        for k_s, v_s in v['book'].items():
            if(len(v_s)!=0 and k_s != "booked"):
                active_dst[k].append([k_s, v_s])
    return active_dst


def get_domains(goal):
    dom = []
    for d, g in goal.items():
        if(len(g)!=0) and d!= "message" and d!= "topic":
            dom.append("MWOZ_"+d)
    return dom


def preprocessGWOZ_template(path_name, translate_path_name, entities_path_name, test_ratio, load_entity=False):
    
    # Get local entities
    if load_entity:
        entity_dic = defaultdict(set)
        entities = json.load(open(entities_path_name))
        for file, value in entities.items():
            for num, v in value.items():
                for head, names in v.items():
                    domain = head.lower().split('-')[0]
                    for name in names:
                        placeholder = 'https://'+domain+'-'+name[0].lower()
                        entity_dic[placeholder].add(name[1])
        
    
    # Get the translations
    id_path_name = '/fsx/waynewu/data/GlobalWOZ/globalwoz_v2/en-source/id.txt' 
    translations = {}
    with open(translate_path_name, 'r') as path_t, open(id_path_name, 'r') as path_i:
        for line_t, line_i in zip(path_t, path_i):
            translations[line_i.rstrip('\n')] = line_t.rstrip('\n')

    # Parse the normal data
    data = []
    dialogue = json.load(open(path_name))
    for i_d, (d_idx, d) in tqdm(enumerate(dialogue.items()),total=len(dialogue.items())):
        dial = {"id":d_idx, "services": get_domains(d['goal']), "dataset":"MWOZ"}
        # if "MWOZ_police" in dial["services"] or "MWOZ_hospital" in dial["services"] or "MWOZ_bus" in dial["services"]: continue
        turns =[]
        dst_prev = {}
        for t_idx, t in enumerate(d['log']):
            if(t_idx % 2 ==0):
                # replace values into placeholders
                span_info_dic = {}
                for counter, slt in enumerate(t["span_info"]):
                    placeholder = 'https://'+slt[0].lower().split('-')[0]+'-'+slt[1].lower()+str(counter)
                    span_info_dic[placeholder] = slt[2].lower()
                    if slt[2] in ['attraction', 'hospital', 'hotel', 'police', 'restaurant', 'taxi', 'train']:
                        t["text"] = t["text"].lower().replace(' '+slt[2].lower(), ' '+placeholder)
                    else:
                        t["text"] = t["text"].lower().replace(slt[2].lower(), placeholder)
                
                # print("USER",t["text"])
                str_API_ACT = ""
                entity_map_api_act = {}
                if "dialog_act" in t:
                    current_api_act_value = []
                    intents_act = set()
                    counter = 0
                    for k,slt in t["dialog_act"].items():
                        # print(k,slt)
                        if "Inform" in k or "Request" in k:
                            domain_name = k.lower().split('-')[0]
                            str_API_ACT += f"{k.lower().replace('-','_')}("
                            for (s,v) in slt:
                                if s != "none" and v != "none":
                                    placeholder = 'https://'+domain_name+'-'+s.lower()
                                    values = entity_dic.get(placeholder, ['none'])
                                    new_value = list(values)[np.random.randint(0, len(values), 1)[0]]
                                    entity_map_api_act[placeholder] = new_value
                                    if load_entity:
                                        str_API_ACT += f'{s.lower()}="{new_value}",'
                                        current_api_act_value.append(new_value)
                                    else:
                                        v = v.replace('"',"'")
                                        str_API_ACT += f'{s.lower()}="{v}",'
                                        current_api_act_value.append(v.replace("'", ""))
                                        counter += 1
                                    # str_API_ACT += f'{k.lower().replace('-','_')}.{s.lower()} = "{v}" '
                                    intents_act.add(k.lower().replace('-','_'))
                            if(str_API_ACT[-1]==","):
                                str_API_ACT = str_API_ACT[:-1]
                            str_API_ACT += ") "
                # print("API", str_API)
                
                ########################### replace placeholders back ###########################
                idd_text = str(d_idx)+'_'+str(t_idx)+'_'+"USER_text"
                idd_span = str(d_idx)+'_'+str(t_idx)+'_'+"USER_span"
                user_text = translations[idd_text]
                user_values =  translations[idd_span].split(',')
                counter = 0
                for placeholder, value in span_info_dic.items():
                    if load_entity:
                        values = entity_dic.get(placeholder[:-1], ['none'])
                        new_value = list(values)[np.random.randint(0, len(values), 1)[0]]
                    else:
                        new_value = user_values[counter] if counter < len(user_values) else value
                    if placeholder in user_text:
                        user_text = user_text.replace(placeholder, new_value)
                    elif '@TAG' in user_text:
                        user_text = user_text.replace('@TAG', new_value)
                    counter += 1
                
                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"USER","utt":user_text, "span_info":span_info_dic})
                
                
            else:
                # replace values into placeholders
                span_info_dic = {}
                for counter, slt in enumerate(t["span_info"]):
                    placeholder = 'https://'+slt[0].lower().split('-')[0]+'-'+slt[1].lower()+str(counter)
                    span_info_dic[placeholder] = slt[2].lower()
                    if slt[2] in ['attraction', 'hospital', 'hotel', 'police', 'restaurant', 'taxi', 'train']:
                        t["text"] = t["text"].lower().replace(' '+slt[2].lower(), ' '+placeholder)
                    else:
                        t["text"] = t["text"].lower().replace(slt[2].lower(), placeholder)
                
                
                api_idd = str(d_idx)+'_'+str(t_idx)+'_'+"API"
                new_values = translations[api_idd].split(',')
                dst_api = get_value_dst(t["metadata"])
                str_API = ""
                current_api_value = []
                intents = set()
                entity_map_api = {}
                counter = 0
                for k,slt in dst_api.items():
                    domain_name = k.lower().split('-')[0]
                    str_API += f"{k.lower().replace('-','_')}("
                    for (s,v) in slt:
                        if len(v)!= 0:
                            placeholder = 'https://'+domain_name+'-'+s.lower()
                            values = entity_dic.get(placeholder, ['none'])
                            new_value = list(values)[np.random.randint(0, len(values), 1)[0]]
                            entity_map_api[placeholder] = new_value
                            if load_entity:
                                str_API += f'{s.lower()}="{new_value}",'
                                current_api_value.append(new_value)
                            else:
                                v = v[0].replace('"',"'")
                                new_value = new_values[counter] if counter < len(new_values) else v
                                # v = translator.translate(v, src=data_type, dest=fewshot_data_type).text
                                str_API += f'{s.lower()}="{new_value}",'
                                current_api_value.append(v.replace("'", ""))
                                counter += 1
                            intents.add(k.lower().replace('-','_'))
                    if(len(str_API)>0 and str_API[-1]==","):
                        str_API = str_API[:-1]
                    str_API += ") "
                if(str_API==""):
                    # use the user dialog act as state
                    turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API","utt":str_API_ACT,"api_act":",".join(current_api_act_value), "service":list(intents_act)})
                    # print("API",str_API_ACT)
                else:
                    turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API","utt":str_API,"api_act":",".join(current_api_value),"service":list(intents)})
                    # print("API", str_API)

                ## API RETURN
                api_idd = str(d_idx)+'_'+str(t_idx)+'_'+"API-OUT"
                new_values = translations[api_idd].split(',')
                str_ACT = ""
                current_act_value = []
                if "dialog_act" in t:
                    counter = 0
                    for k,slt in t["dialog_act"].items():
                        # print(k,slt)
                        if "Inform" in k or "Recommend" in k or "Booking-Book" in k or "-Select" in k:
                            domain_name = k.lower().split('-')[0]
                            str_ACT += f"{k.lower().replace('-','_')}("
                            for (s,v) in slt:
                                if s != "none" and v != "none":
                                    placeholder = 'https://'+domain_name+'-'+s.lower()
                                    if placeholder in entity_map_api:
                                        new_value = entity_map_api[placeholder]
                                    else:
                                        values = entity_dic.get(placeholder, ['none'])
                                        new_value = list(values)[np.random.randint(0, len(values), 1)[0]]
                                        entity_map_api[placeholder] = new_value
                                    if load_entity:
                                        str_ACT += f'{s.lower()}="{new_value}",'
                                        current_act_value.append(new_value)
                                    else:
                                        v = v.replace('"',"'")
                                        new_value = new_values[counter] if counter < len(new_values) else v
                                        # v = translator.translate(v, src=data_type, dest=fewshot_data_type).text
                                        str_ACT += f'{s.lower()}="{new_value}",'
                                        current_act_value.append(v.replace("'", ""))
                                        # str_ACT += f'{k.lower().replace("-",".")}.{s.lower()} = "{v}" '
                                        counter += 1
                            if(str_ACT[-1]==","):
                                str_ACT = str_ACT[:-1]
                            str_ACT += ") "
                        if "Booking-NoBook" in k:
                            # str_ACT += f'{k.lower().replace("-",".")} '
                            str_ACT += f"{k.lower().replace('-','_')}() "
                
                ########################### replace placeholders back ###########################
                idd_text = str(d_idx)+'_'+str(t_idx)+'_'+"SYSTEM_text"
                idd_span = str(d_idx)+'_'+str(t_idx)+'_'+"SYSTEM_span"
                system_text = translations[idd_text]
                system_values =  translations[idd_span].split(',')
                counter = 0
                for placeholder, value in span_info_dic.items():
                    if load_entity:
                        new_value = entity_map_api[placeholder[:-1]]
                    else:
                        new_value = system_values[counter] if counter < len(system_values) else value
                    if placeholder in system_text:
                        system_text = system_text.replace(placeholder, new_value)
                    elif '@TAG' in system_text:
                        system_text = system_text.replace('@TAG', new_value)
                    counter += 1

                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API-OUT","utt":str_ACT,"api_act":",".join(current_act_value),"service":None})
                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"SYSTEM","utt":system_text,"api_act":"","span_info":span_info_dic})
        
        dial["dialogue"] = turns
        for t in turns:
            for k,v in t.items():
                print(k, ": ", v)
            print('-------')
        dafe
        data.append(dial)

    # Split the data into train/dev/test
    id_list = [dial["id"] for dial in data]
    length = len(id_list)
    train_id = id_list[:int(length*0.8)]
    dev_id = id_list[int(length*0.8):int(length*test_ratio)]
    test_id = id_list[int(length*test_ratio):]

    train_data, dev_data, test_data = [], [], []

    for dial in data:
        if dial["id"] in dev_id:
           dev_data.append(dial)
        elif dial["id"] in test_id:
           test_data.append(dial)
        else:
           train_data.append(dial)
    return train_data, dev_data, test_data