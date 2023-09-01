import glob
import json
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


def preprocessGWOZ(path_name, test_ratio, seed):
    
    # cache paths
    prefix = '/'.join(path_name.split('/')[:-1])
    train_path = prefix + '/train.pkl'
    dev_path = prefix + '/dev.pkl'
    test_path = prefix + '/test.pkl'
    if (os.path.exists(train_path) and os.path.exists(dev_path) and os.path.exists(test_path)):
        print('Loading cached data directly...')
        train_data = pickle.load(open(train_path, 'rb'))
        dev_data = pickle.load(open(dev_path, 'rb'))
        test_data = pickle.load(open(test_path, 'rb'))
        
        return train_data, dev_data, test_data
    
    data = []
    dialogue = json.load(open(path_name))
    for i_d, (d_idx, d) in tqdm(enumerate(dialogue.items()),total=len(dialogue.items())):
        dial = {"id":d_idx, "services": get_domains(d['goal']), "dataset":"MWOZ"}
        # if "MWOZ_police" in dial["services"] or "MWOZ_hospital" in dial["services"] or "MWOZ_bus" in dial["services"]: continue
        turns =[]
        dst_prev = {}
        for t_idx, t in enumerate(d['log']):
            if(t_idx % 2 ==0):
                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"USER","utt":t["text"]})
                # print("USER",t["text"])
                str_API_ACT = ""
                if "dialog_act" in t:
                    intents_act = set()
                    for k,slt in t["dialog_act"].items():
                        # print(k,slt)
                        if "Inform" in k or "Request" in k:
                            str_API_ACT += f"{k.lower().replace('-','_')}("
                            for (s,v) in slt:
                                if s != "none" and v != "none":
                                    v = v.replace('"',"'")
                                    str_API_ACT += f'{s.lower()}="{v}",'
                                    # str_API_ACT += f'{k.lower().replace('-','_')}.{s.lower()} = "{v}" '
                                    intents_act.add(k.lower().replace('-','_'))
                            if(str_API_ACT[-1]==","):
                                str_API_ACT = str_API_ACT[:-1]
                            str_API_ACT += ") "
                # print("API", str_API)
            else:
                dst_api = get_value_dst(t["metadata"])
                str_API = ""
                intents = set()
                for k,slt in dst_api.items():
                    str_API += f"{k.lower().replace('-','_')}("
                    for (s,v) in slt:
                        if len(v)!= 0:
                            v = v[0].replace('"',"'")
                            str_API += f'{s.lower()}="{v}",'
                            intents.add(k.lower().replace('-','_'))
                    if(len(str_API)>0 and str_API[-1]==","):
                        str_API = str_API[:-1]
                    str_API += ") "
                if(str_API==""):
                    turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API","utt":str_API_ACT,"service":list(intents_act)})
                    # print("API",str_API_ACT)
                else:
                    turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API","utt":str_API,"service":list(intents)})
                    # print("API", str_API)

                ## API RETURN
                str_ACT = ""
                if "dialog_act" in t:
                    for k,slt in t["dialog_act"].items():
                        # print(k,slt)
                        if "Inform" in k or "Recommend" in k or "Booking-Book" in k or "-Select" in k:
                            str_ACT += f"{k.lower().replace('-','_')}("
                            for (s,v) in slt:
                                if s != "none" and v != "none":
                                    v = v.replace('"',"'")
                                    str_ACT += f'{s.lower()}="{v}",'
                                    # str_ACT += f'{k.lower().replace("-",".")}.{s.lower()} = "{v}" '
                            if(str_ACT[-1]==","):
                                str_ACT = str_ACT[:-1]
                            str_ACT += ") "
                        if "Booking-NoBook" in k:
                            # str_ACT += f'{k.lower().replace("-",".")} '
                            str_ACT += f"{k.lower().replace('-','_')}() "

                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API-OUT","utt":str_ACT,"service":None})
                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"SYSTEM","utt":t["text"]})
        dial["dialogue"] = turns
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