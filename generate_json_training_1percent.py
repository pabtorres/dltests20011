import json
d = {}
f = open('mapping_imagenet.txt',)
for line in f:
  u = line.split(" ")
  num, index, _, _ = u
  d[num] = index

f = open('1percent.txt', )
training_data_list = []
folder = "1percentImageNet"
folder_2 = "1percentimagenet"
for line in f:
  u = line.split("_")
  file_name = line[:-1]
  image_folder_name, _ = u
  label = d[image_folder_name]
  file_dir = folder + '/' + folder_2 + '/' + file_name
  training_data_list.append((file_dir, label))

with open('./training_data_list_1percent.json', 'w') as f:
  json.dump(training_data_list, f)