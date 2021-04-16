from pathlib import Path

with open('train.txt', 'w') as train_file:
    with open('val.txt', 'w') as val_file:
        for folder in Path('filelists').iterdir():
            for data in folder.iterdir():
                if int(str(folder).split('/')[-1]) < 60:
                    train_file.write(str(data)[str(data).index('/')+1:] + '\n')
                else:
                    val_file.write(str(data)[str(data).index('/')+1:] + '\n')

