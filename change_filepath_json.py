import json
import os
# wav_dir = "/work/hdd/bebr/DATA-LB_audio-old/LB_2mins_files/wav"
wav_dir = "/work/hdd/bebr/PRJ_LLM_SP25/data/wav_mix"
# json_file_folder = "/u/mkhan14/LittleBeats-LLM/nur_json_mix_LB_only"
# json_list = os.listdir(json_file_folder)
json_file = "/work/hdd/bebr/PRJ_LLM_SP25/LittleBeats-LLM/json_event_based/train_snr_10_mix_8_10s.json"

splitname = json_file.split('/')[-1].split('_')[0]
duration = json_file.split('/')[-1].split('_')[-1].split('.')[0]
dirname = json_file.split('/')[-1][5:29]
dirname= json_file.split('/')[-1][5:-5]

# final_wav_dir = os.path.join(wav_dir, dirname, splitname)
# for json_file in json_list:
#     splitname = json_file.split('/')[-1].split('_')[0]
#     duration = json_file.split('/')[-1].split('_')[-1].split('.')[0]
#     if splitname=='dev' and (duration =='2s' or duration =='5s'):
#         dirname = json_file.split('/')[-1][4:-8]
#     elif splitname=='dev' and duration =='10s':
#         dirname = json_file.split('/')[-1][4:-9]
#     elif splitname=='test' and (duration =='2s' or duration =='5s'):
#         dirname = json_file.split('/')[-1][5:-8]
#     elif splitname=='test' and duration =='10s':
#         dirname = json_file.split('/')[-1][5:-9]
#     elif splitname=='train' and (duration =='2s' or duration =='5s'):
#         dirname = json_file.split('/')[-1][6:-8]
#     elif splitname=='train' and duration =='10s':
#         dirname = json_file.split('/')[-1][6:-9]
    

# dirname2 = dirname[:-1]+'40'
# final_wav_dir = os.path.join(wav_dir, dirname, splitname)
# final_wav_dir2 = os.path.join(wav_dir, dirname2, splitname)




# wav_list = os.listdir(wav_dir)

# print(dirname, dirname2, final_wav_dir, final_wav_dir2)

# json_file = os.path.join(json_file_folder, json_file)
with open(json_file, 'r') as file:
    data_json = json.load(file)
data_keys = data_json.keys()

final_wav_dir = "/work/hdd/bebr/PRJ_LLM_SP25/data/wav_mix/snr_10_mix_40/train"
wav_list = os.listdir(final_wav_dir)
final_wav_dir2 = "/work/hdd/bebr/PRJ_LLM_SP25/data/wav_mix/snr_10_mix_8/train"
wav_list2 = os.listdir(final_wav_dir2)
# print(wav_list)
count = 0
count2 =0
for key in data_keys:
    file = data_json[key]['wav']['file']
    file = file.split('/')[-1]
    if file in wav_list:
        data_json[key]['wav']['file'] = os.path.join(final_wav_dir, file)
        count2+=1
    elif file in wav_list2:
        data_json[key]['wav']['file'] = os.path.join(final_wav_dir2, file)
    else:
        # print("not found")
        count +=1
print(count, count2)
    # elif file in wav_list2:
    #     data_json[key]['wav']['file'] = os.path.join(final_wav_dir2, file)
            # new_keys = {}
        # new_keys['wav'] = {}
        # new_keys['wav']['file'] = os.path.join(final_wav_dir, file)
        # new_keys['wav']['start'] = data_json[key]['wav']['start']
        # new_keys['wav']['stop'] = data_json[key]['wav']['stop']
        # new_keys['label'] = {}
        # label_key = list(data_json[key]['label'].keys())[0]
        # new_keys['label'][label_key] = data_json[key]['label'][label_key]
        # print(new_keys)
        # break

save_name = os.path.join('nur_json_mix_LB_only/', json_file.split('/')[-1])
print(save_name)
with open(save_name, 'w') as f:
    json.dump(data_json, f, indent=4)