import os
path = '/home/shreeyash/Desktop/noise_Dataset/'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, 'noise'+ str(i)+'.jpg'))
    i = i+1