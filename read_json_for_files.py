import os
import json

json_file = "/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/nur_json_mix_LB_only/test_2s_real.json"
with open(json_file, 'r') as file:
    data_json = json.load(file)
data_keys = data_json.keys()

visit_name = {'ACRI': {}, 'CCHMC': {}, 'Cedars': {}, 'NYU': {}, 'OHSU': {}}
for key in data_keys:
    visit = key.split('_')[0]
    id = key.split('_')[1]
    if id in visit_name[visit]:
        visit_name[visit][id] +=1
    else:
        visit_name[visit][id] = 1

print(visit_name)