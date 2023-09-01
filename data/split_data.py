import json

paths = ['/fsx/waynewu/data/GlobalWOZ/globalwoz_v2/en-source/multiwoz.json',
         '/fsx/waynewu/data/GlobalWOZ/globalwoz_v2/es/F&F_es.json',
         '/fsx/waynewu/data/GlobalWOZ/globalwoz_v2/id/F&F_id.json',
         '/fsx/waynewu/data/GlobalWOZ/globalwoz_v2/zh/F&F_zh.json']
folders = ['en-source', 'es', 'id', 'zh']

def split_data(path, folder):
    print('Running on: ', path)
    dic = json.load(open(path, 'r'))
    length = len(dic.keys())
    dic_list = list(dic.items())
    
    train = dict(dic_list[:int(length*0.8)])
    valid = dict(dic_list[int(length*0.8):int(length*0.9)])
    test = dict(dic_list[int(length*0.9):])
    
    json.dump(train, open('/fsx/waynewu/data/GlobalWOZ/globalwoz_v2/{}/train.json'.format(folder), 'w'))
    json.dump(valid, open('/fsx/waynewu/data/GlobalWOZ/globalwoz_v2/{}/valid.json'.format(folder), 'w'))
    json.dump(test, open('/fsx/waynewu/data/GlobalWOZ/globalwoz_v2/{}/test.json'.format(folder), 'w'))

for path, folder in zip(paths, folders):
    split_data(path, folder)

