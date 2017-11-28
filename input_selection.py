import numpy as np
import os
import glob
import shutil
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.getcwd()
INPUT_FOLDER = '/home/senpai/Documents/Github/Ten_Ton_Talker/Input_audio_wav_16k/'
talkers = os.listdir(INPUT_FOLDER)
talkers.sort()

audio_dict = {}

for l in talkers:
    audio_dict[l] = sorted(os.listdir(INPUT_FOLDER + l))

random_audioList = sorted(np.random.choice(audio_dict['AGNES'], 50, replace=False))

trimmed_audioList = []
for i in random_audioList:
    trimmed_audioList.append(i.split('_')[0])
#
# x_train, y_test = train_test_split(trimmed_audioList, test_size=0.1, random_state=42)
#
# txtfile = open('./test_set.txt', mode='w')
# for x in sorted(y_test):
#     txtfile.write(x + "\n")
# txtfile.close()
#
# x_train, y_val = train_test_split(x_train, test_size=0.2, random_state=42)

txtfile = open('./train_set.txt', mode='w')
for x in sorted(trimmed_audioList):
    txtfile.write(x + "\n")
txtfile.close()
#
# txtfile = open('./val_set.txt', mode='w')
# for x in sorted(y_val):
#     txtfile.write(x + "\n")
# txtfile.close()

training_pathList = []
for i in talkers:
    for j in trimmed_audioList:
        for filename in glob.iglob(INPUT_FOLDER + i + '/' + j + '*', recursive=True):
            training_pathList.append(filename)

destination = '/home/senpai/Documents/Github/ContradictiveNeuralNet/Input_audio/Train/'
for files in sorted(training_pathList):
    shutil.copy(files, destination)


# val_pathList = []
# for i in talkers:
#     for j in y_val:
#         for filename in glob.iglob(INPUT_FOLDER + i + '/' + j + '*', recursive=True):
#             val_pathList.append(filename)
#
# destination = '/home/senpai/Documents/Github/ContradictiveNeuralNet/Input_audio/Val/'
# for files in sorted(val_pathList):
#     shutil.copy(files, destination)
#
#
# test_pathList = []
# for i in talkers:
#     for j in y_test:
#         for filename in glob.iglob(INPUT_FOLDER + i + '/' + j + '*', recursive=True):
#             test_pathList.append(filename)
#
# destination = '/home/senpai/Documents/Github/ContradictiveNeuralNet/Input_audio/Test/'
# for files in sorted(test_pathList):
#     shutil.copy(files, destination)
