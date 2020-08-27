import os
import shutil

import librosa

source_path = '../database/blizzard2013/segmented/wavn'
dest_path = '../database/blizzard2013/segmented/small_wavn_lead_trail_silence_removed_16000'
data_length = 7200  # sec

############################
text_dict = {}
text_file = open('../database/blizzard2013/segmented/metadata.csv', 'r')
Lines = text_file.readlines()
for line in Lines:
    a = line.rstrip().split('|')
    text_dict[a[0]] = a[-1]
text_file.close()

#########################
if os.path.isdir(dest_path):
    shutil.rmtree(dest_path)
os.mkdir(dest_path)

small_text_file = open(dest_path + '/metadata.csv', 'w')
dur = 0
file_list = os.listdir(source_path)
for f in file_list:
    if f.endswith('.wav'):
        os.system('sox ' + source_path + '/' + f
                  + ' -r 16000 -c 1 -b 16 ' + dest_path + '/' + f
                  + ' silence -l 1 0.1 0.4% -1 0.2 1%')
        x, y = librosa.load(dest_path + '/' + f)
        dur = dur + len(x) / y

        file_name = f.replace('.wav', '')
        small_text_file.write(file_name + '|' + text_dict[file_name] + "\n")
        if dur > data_length:
            break
        print(dur)
small_text_file.close()
