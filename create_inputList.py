import os

TRAINING_FOLDER = 'Input_spectrogram/Training/'

input_list = sorted(os.listdir(TRAINING_FOLDER))

txtfile = open('./img_set_train.txt', mode='w')

for i in input_list:
    txtfile.write(i + " " + i.split('_')[0])
    txtfile.write("\n")
txtfile.close()


VALIDATION_FOLDER = 'Input_spectrogram/Validation/'

input_list = sorted(os.listdir(VALIDATION_FOLDER))

txtfile = open('./img_set_val.txt', mode='w')

for i in input_list:
    txtfile.write(i + " " + i.split('_')[0])
    txtfile.write("\n")
txtfile.close()
