from glob import glob

root_path = f'/workspace/LivDet2023/data/Train_LivDet2023'
dataset_name = 'Dermalog'

for kind in (['Fake', 'Live']):
    for path in glob(f'{root_path}/{dataset_name}/**/'):
        print(kind)
        print(path)
