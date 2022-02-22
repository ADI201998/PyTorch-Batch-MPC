import os
path = '/home/aditya/Documents/LSTR-custom/LSTR/raws/train_labels/'
files = os.listdir(path)
print(files)

for index, file in enumerate(files):
    print(index, file)
    #os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index-1), '.jpg'])))
