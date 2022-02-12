import os

def getDataSet(dir_path):
    data_sets = []    

    sub_dirs = os.listdir(dir_path)
    for classId in sub_dirs:
        sub_dir_path = dir_path + '/' + classId
        img_files = os.listdir(sub_dir_path)
        for f in img_files:
            data_sets.append([classId, sub_dir_path + '/' + f])

    return data_sets

if __name__ == '__main__':
    print('use for import')