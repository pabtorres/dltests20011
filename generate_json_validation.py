import json

def padding(input,padding=8):
  zeros_to_add = padding - len(str(input))
  return f'ILSVRC2012_val_{zeros_to_add*"0"+str(input)}.JPEG'

d = {}
f = open('mapping_imagenet.txt',)
for line in f:
  u = line.split(" ")
  num, index_standard, index_imagenet, _ = u
  d[index_imagenet] = index_standard

f = open('ILSVRC2012_validation_ground_truth.txt', )
validation_data_list = []
folder = "validationImageNet"
image_number = 1
for line in f:
  index_imagenet = line[:-1]
  file_name = padding(image_number)
  image_number+=1
  standard_label = d[index_imagenet]
  file_dir = folder + '/' + file_name
  validation_data_list.append((file_dir, standard_label))

with open('./validation_data_list.json', 'w') as f:
  json.dump(validation_data_list, f)