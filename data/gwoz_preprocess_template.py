import glob
import json
import torch
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import os
import pickle

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


def preprocessGWOZ_template(path_name, translate_path_name, data_type, fewshot_data_type, test_ratio, preload=False, seed=None, enable_fe=True):
    
    # cache paths
    prefix = '/'.join(path_name.split('/')[:-1]) + '/translate'
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    
    train_path = prefix + '/train_translate_{}.pkl'.format(fewshot_data_type)
    dev_path = prefix + '/dev_translate_{}.pkl'.format(fewshot_data_type)
    test_path = prefix + '/test_translate_{}.pkl'.format(fewshot_data_type)
    if (os.path.exists(train_path) and os.path.exists(dev_path) and os.path.exists(test_path)):
        print('Loading cached data directly...')
        train_data = pickle.load(open(train_path, 'rb'))
        dev_data = pickle.load(open(dev_path, 'rb'))
        test_data = pickle.load(open(test_path, 'rb'))
        
        return train_data, dev_data, test_data
    
    if enable_fe:
        print('Remain the English entities...')
    else:
        print('Replace entities with ', fewshot_data_type)
    
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
                if "dialog_act" in t:
                    current_api_act_value = []
                    intents_act = set()
                    counter = 0
                    for k,slt in t["dialog_act"].items():
                        # print(k,slt)
                        if "Inform" in k or "Request" in k:
                            str_API_ACT += f"{k.lower().replace('-','_')}("
                            for (s,v) in slt:
                                if s != "none" and v != "none":
                                    v = v.replace('"',"'")
                                    str_API_ACT += f'{s.lower()}="{v}",'
                                    current_api_act_value.append(v.replace("'", ""))
                                    # str_API_ACT += f'{k.lower().replace('-','_')}.{s.lower()} = "{v}" '
                                    intents_act.add(k.lower().replace('-','_'))
                                    counter += 1
                            if(str_API_ACT[-1]==","):
                                str_API_ACT = str_API_ACT[:-1]
                            str_API_ACT += ") "
                # print("API", str_API)
                
                # replace placeholders back
                idd_text = str(d_idx)+'_'+str(t_idx)+'_'+"USER_text"
                idd_span = str(d_idx)+'_'+str(t_idx)+'_'+"USER_span"
                user_text = translations[idd_text]
                user_values =  translations[idd_span].split(',')
                counter = 0
                for placeholder, value in span_info_dic.items():
                    #************************** replace the template value **************************
                    if enable_fe:
                        new_value = value
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
                counter = 0
                for k,slt in dst_api.items():
                    str_API += f"{k.lower().replace('-','_')}("
                    for (s,v) in slt:
                        if len(v)!= 0:
                            v = v[0].replace('"',"'")
                            #************************** replace the template value **************************
                            if enable_fe:
                                new_value = v
                            else:
                                new_value = new_values[counter] if counter < len(new_values) else v
                            # v = translator.translate(v, src=data_type, dest=fewshot_data_type).text
                            str_API += f'{s.lower()}="{new_value}",'
                            current_api_value.append(v.replace("'", ""))
                            intents.add(k.lower().replace('-','_'))
                            counter += 1
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
                            str_ACT += f"{k.lower().replace('-','_')}("
                            for (s,v) in slt:
                                if s != "none" and v != "none":
                                    v = v.replace('"',"'")
                                    #************************** replace the template value **************************
                                    if enable_fe:
                                        new_value = v
                                    else:
                                        new_value = new_values[counter] if counter < len(new_values) else v
                                    new_value = v
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
                
                # replace placeholders back
                idd_text = str(d_idx)+'_'+str(t_idx)+'_'+"SYSTEM_text"
                idd_span = str(d_idx)+'_'+str(t_idx)+'_'+"SYSTEM_span"
                system_text = translations[idd_text]
                system_values =  translations[idd_span].split(',')
                counter = 0
                for placeholder, value in span_info_dic.items():
                    #************************** replace the template value **************************
                    # new_value = system_values[counter] if counter < len(system_values) else value
                    new_value = value
                    if placeholder in system_text:
                        system_text = system_text.replace(placeholder, new_value)
                    elif '@TAG' in system_text:
                        system_text = system_text.replace('@TAG', new_value)
                    counter += 1

                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API-OUT","utt":str_ACT,"api_act":",".join(current_act_value),"service":None})
                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"SYSTEM","utt":system_text,"api_act":"","span_info":span_info_dic})
        
        dial["dialogue"] = turns
        # for t in turns:
        #     for k,v in t.items():
        #         print(k, ": ", v)
        #     print('-------')
        # dafe
        data.append(dial)

    # shuffle the ids
    np.random.seed(seed)
    id_list = np.array([dial["id"] for dial in data])
    np.random.shuffle(id_list)
    
    # Split the data into train/dev/test
    length = len(id_list)
    train_id = id_list[:int(length*0.8)]
    dev_id = id_list[int(length*0.8):int(length*0.9)]
    test_id = id_list[int(length*test_ratio):] # only take the subset for fast evalaution

    train_data, dev_data, test_data = [], [], []

    for dial in data:
        if dial["id"] in dev_id:
           dev_data.append(dial)
        elif dial["id"] in test_id:
           test_data.append(dial)
        else:
           train_data.append(dial)
    
    # cache processed data
    pickle.dump(train_data, open(train_path, 'wb'))
    pickle.dump(dev_data, open(dev_path, 'wb'))
    pickle.dump(test_data, open(test_path, 'wb'))
    
    return train_data, dev_data, test_data