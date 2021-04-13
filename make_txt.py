from pathlib import Path

with open('train.txt', 'w') as train_file:
    with open('val.txt', 'w') as val_file:
        for folder in Path('filelists').iterdir():
            if int(str(folder).split('/')[-1]) < 60:
                train_file.write('filelists/' + str(folder).split('/')[-1] + '\n')
            else:
                val_file.write('filelists/' + str(folder).split('/')[-1] + '\n')

