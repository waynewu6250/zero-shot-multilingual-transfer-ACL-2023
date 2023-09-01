"""Evaluation functions"""

import json
from config import opt
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import subprocess
import tempfile
import re
from nltk.translate.bleu_score import sentence_bleu
from torchmetrics import SacreBLEUScore
from dictdiffer import diff

def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score

def cal_kl(data, data_len, save_path, type=None):
        prob_t, prob_p = [0]*data_len, [0]*data_len
        for dt, dp in data:
            prob_t[dt] += 1
            prob_p[dp] += 1
        prob_t = torch.Tensor(prob_t)/len(data)
        prob_p = torch.Tensor(prob_p)/len(data)

        # Plot KL distribution
        plt.figure(figsize=(20,10))
        plt.plot(np.arange(len(prob_t)), prob_t, label='true distribution', linewidth=2.0)
        plt.plot(np.arange(len(prob_p)), prob_p, label='predicted distribution', linewidth=2.0)
        output_path = os.path.join(save_path, 'KL_div_{}.jpg'.format(type))
        
        plt.legend()
        plt.grid()
        label_fontsize = 30
        plt.title('True and Predicted field distribution', fontsize=30)
        plt.xlabel('Class', fontsize=label_fontsize)
        plt.ylabel('Probability', fontsize=label_fontsize)
        plt.tick_params(axis='both', which='major', labelsize=label_fontsize)
        plt.legend(prop={"size":25})
        plt.savefig(output_path)

        prob_t[prob_t==0] = 1e-10
        prob_p[prob_p==0] = 1e-10
        return (prob_t * (prob_t / prob_p).log()).sum()

###########################################################################################
def parse_API(text):
    API = defaultdict(lambda:defaultdict(str))
    for function in text.split(") "):
        if(function!=""):
            if("(" in function and len(function.split("("))==2):
                intent, parameters = function.split("(")
                parameters = sum([s.split('",') for s in parameters.split("=")],[])
                if len(parameters)>1:
                    if len(parameters) % 2 != 0:
                        parameters = parameters[:-1]

                    for i in range(0,len(parameters),2):
                        API[intent][parameters[i]] = parameters[i+1].replace('"',"")

                if(len(API)==0): API[intent]["none"] = "none"
    return API
    
def evaluate_EER(results_dict):
    ERR = []
    cnt_bad = 0
    cnt_superflous = 0
    tot = 0
    
    for d in results_dict:
        if(d["spk"]=="SYSTEM"):
            ent = set()
            ent_corr = []
            d['turn_belief'] = d['turn_belief'].split("API-OUT: ")[1]
            if(d['turn_belief']==""):
                continue

            for speech_act, slot_value_dict in parse_API(d['turn_belief']+" ").items():
                tot += len(slot_value_dict.keys())
                for s,v in slot_value_dict.items():
                    if(v not in ["True", "False", "yes", "no", "?","none"]):
                        if(v.lower() not in d["genr"].lower()):
                            cnt_bad += 1
                        else:
                            ent_corr.append(v.lower())
                        ent.add(v.lower())

    return (cnt_bad+cnt_superflous)/float(tot)


def evaluate_API(pred,gold):
    intent_accuracy = []
    turn_level_slot_acc = []
    turn_level_joint_acc = []
    for p, g in zip(pred,gold):
        API_G = {}
        API_P = {}
        # Add a space since the gold has a space left after removing [eos] or </s>
        p = p+" "
        if(g!=""):
            API_G = parse_API(g)
            # print(API_G)
            if(p!="" and "(" in p and ")"): ## means the predicted text is an API
                API_P = parse_API(p)
                if len(API_G.keys()) != 1: 
                    continue
                if len(API_P.keys()) != 1: 
                    turn_level_joint_acc.append(0)
                    continue
                # intent accuracy
                intent_G = list(API_G.keys())[0]
                intent_P = list(API_P.keys())[0]
                if(intent_G==intent_P):
                    intent_accuracy.append(1)
                else:
                    intent_accuracy.append(0)

                state_G = {s:v for s,v in API_G[intent_G].items() if s !="none"}
                state_P = {s:v for s,v in API_P[intent_P].items() if s !="none"}
                
                if(len([d for d in diff(state_G,state_P)])==0):
                    turn_level_joint_acc.append(1)
                else:
                    turn_level_joint_acc.append(0)

            else:
                intent_accuracy.append(0)
                turn_level_joint_acc.append(0)
                turn_level_slot_acc.append(0)

    return np.mean(intent_accuracy), np.mean(turn_level_joint_acc)


def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score for hypotheses and references
    using the MOSES ulti-bleu.perl script.
    Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script
    Returns:
    The BLEU score as a float32 value.
    """

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    multi_bleu_path = "./multi-bleu.perl"
    os.chmod(multi_bleu_path, 0o755)


    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()


     # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print("multi-bleu.perl script returned non-zero exit code")
                print(error.output)
                bleu_score = np.float32(0.0)

    # Close temp files
    hypothesis_file.close()
    reference_file.close()
    return bleu_score

def cal_rg(results, datause='13a'):

    domain_DST = []
    domain_BLEU = defaultdict(list)
    domain_API = defaultdict(list)
    bleu_score1, bleu_score2, total_num, EER = 0, 0, 0, 0
    bleu = SacreBLEUScore(tokenize=datause if datause == 'zh' else '13a')
    for r in results:
        if(r['spk']=='SYSTEM'):
            # Response generation: BLEU
            domain_BLEU["pred"].append(r['genr'].strip())
            domain_BLEU["gold"].append([r['gold'].strip()])
            bleu_score1 += sentence_bleu([r['gold'].strip()], r['genr'].strip(), weights=(1.0, 0.0, 0.0, 0.0))
            bleu_score2 += sentence_bleu([r['gold'].strip()], r['genr'].strip(), weights=(0.0, 1.0, 0.0, 0.0))
            total_num += 1

            # Response generation: EER
            domain_DST.append(r)

        elif(r['spk']=='API'):
            # DST task
            domain_API["pred"].append(r['genr'])
            domain_API["gold"].append(r['gold'])
    
    # Response generation
    # BLEU = moses_multi_bleu(domain_BLEU["pred"], domain_BLEU["gold"])
    EER = evaluate_EER(domain_DST)
    BLEU_score = bleu(domain_BLEU["pred"], domain_BLEU["gold"])

    # DST task
    intent_accuracy, turn_level_joint_acc = evaluate_API(domain_API["pred"], domain_API["gold"])


    return {'BLEU_score': BLEU_score, 
            'BLEU1': bleu_score1/total_num, 
            'BLEU2': bleu_score2/total_num, 
            'SER': EER,
            'turn_level_joint_acc': turn_level_joint_acc
            }


    