import os
import random
import shutil

test_rate = 0.1
path_name='D:/NotOnlyCode/DIP/downloads'
dir = ['train','val','test']
symbol = ['healthy', 'leaf_rust', 'septoria']

# rename pactures in every child file
def rename_everyfile_image():
    def rename(path, t):
        i = 0
        path = path.replace("/",'\\')
        for item in os.listdir(path):
            try:
                os.rename(os.path.join(path, item), os.path.join(path, (t+str(i)+'.jpg'))) #os.path.join(path_name,item)表示找到每个文件的绝对路径并进行拼接操作
            except(FileExistsError):
                return
            finally:
                i += 1

    for i in range(0,3):
        tmp = os.path.join(path_name, dir[i])
        for j in range(0,3):
            cur_path = os.path.join(tmp, symbol[j])
            # print(cur_path)
            if(dir[i]=='test'):
                continue
            else:
                rename(cur_path, symbol[j])

#exchange image
def exchange_image(train_path, val_path):
    train_image_path = []
    val_image_path = []

    for item in os.listdir(train_path):
        train_image_path.append(item)
    for item in os.listdir(val_path):
        val_image_path.append(item)
    
    exchange_num = len(val_image_path)
    print("NUM:", exchange_num)
    sample = random.sample(train_image_path, exchange_num)

    print(sample)
    for i in sample:
        shutil.copy()

    return

if __name__ == "__main__":
    # rename_everyfile_image()

    for i in range(0, 3):
        train_path = os.path.join(path_name, 'train', symbol[i]).replace("/",'\\')
        val_path = os.path.join(path_name, 'val', symbol[i]).replace("/",'\\')
        exchange_image(train_path, val_path)
        break




