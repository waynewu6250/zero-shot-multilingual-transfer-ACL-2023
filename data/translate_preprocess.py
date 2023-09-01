import glob
import json
import torch
from collections import defaultdict
from tqdm import tqdm

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

def preprocessGWOZ_template(path_name):
    """Preprocess GlobalWOZ data

    Parse states from API given the current SYSTEM metadata; if none, use the USER dialog act in the previous turn.
    Parse system acts from API given the current SYSTEM dialog act.
    """
    
    id_f = open('/fsx/waynewu/data/GlobalWOZ/globalwoz_v2/en-source/id.txt', 'w')
    utt_f = open('/fsx/waynewu/data/GlobalWOZ/globalwoz_v2/en-source/utt.txt', 'w')
    
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
                
                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"USER","utt":t["text"], "span_info":span_info_dic})
                # print("USER",t["text"])
                str_API_ACT = ""
                if "dialog_act" in t:
                    current_api_act_value = []
                    intents_act = set()
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
                            if(str_API_ACT[-1]==","):
                                str_API_ACT = str_API_ACT[:-1]
                            str_API_ACT += ") "
                # print("API", str_API)
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
                
                dst_api = get_value_dst(t["metadata"])
                str_API = ""
                current_api_value = []
                intents = set()
                for k,slt in dst_api.items():
                    str_API += f"{k.lower().replace('-','_')}("
                    for (s,v) in slt:
                        if len(v)!= 0:
                            v = v[0].replace('"',"'")
                            # v = translator.translate(v, src=data_type, dest=fewshot_data_type).text
                            str_API += f'{s.lower()}="{v}",'
                            current_api_value.append(v.replace("'", ""))
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
                str_ACT = ""
                current_act_value = []
                if "dialog_act" in t:
                    for k,slt in t["dialog_act"].items():
                        # print(k,slt)
                        if "Inform" in k or "Recommend" in k or "Booking-Book" in k or "-Select" in k:
                            str_ACT += f"{k.lower().replace('-','_')}("
                            for (s,v) in slt:
                                if s != "none" and v != "none":
                                    v = v.replace('"',"'")
                                    # v = translator.translate(v, src=data_type, dest=fewshot_data_type).text
                                    str_ACT += f'{s.lower()}="{v}",'
                                    current_act_value.append(v.replace("'", ""))
                                    # str_ACT += f'{k.lower().replace("-",".")}.{s.lower()} = "{v}" '
                            if(str_ACT[-1]==","):
                                str_ACT = str_ACT[:-1]
                            str_ACT += ") "
                        if "Booking-NoBook" in k:
                            # str_ACT += f'{k.lower().replace("-",".")} '
                            str_ACT += f"{k.lower().replace('-','_')}() "

                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"API-OUT","utt":str_ACT,"api_act":",".join(current_act_value),"service":None})
                turns.append({"dataset":"MWOZ","id":d_idx,"turn_id":t_idx,"spk":"SYSTEM","utt":t["text"],"api_act":"","span_info":span_info_dic})
            
            # for t in turns:
            #     for k,v in t.items():
            #         print(k, ": ", v)
            #     print('-------')
            # dafe
            data.append(turns)
        for turn in turns:
            if turn['spk'] == 'USER' or turn['spk'] == 'SYSTEM':
                id_f.write(str(turn['id'])+'_'+str(turn['turn_id'])+'_'+turn['spk']+'_text'+'\n')
                utt_f.write(turn['utt']+'\n')
                id_f.write(str(turn['id'])+'_'+str(turn['turn_id'])+'_'+turn['spk']+'_span'+'\n')
                code = '@TAG' if len(turn['span_info']) == 0 else ','.join(turn['span_info'].values())
                utt_f.write(code+'\n')
            else:
                id_f.write(str(turn['id'])+'_'+str(turn['turn_id'])+'_'+turn['spk']+'\n')
                code = '@TAG' if turn['api_act'] == '' else turn['api_act']
                utt_f.write(code+'\n')
        
    id_f.close()
    utt_f.close()

path_name = '/fsx/waynewu/data/GlobalWOZ/globalwoz_v2/en-source/multiwoz.json'
preprocessGWOZ_template(path_name)