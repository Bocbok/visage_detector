from util import *
from functools import partial
from tqdm import tqdm

# Create the dataset. Need to run the func once for the train and once for validation


if __name__ == '__main__':

    folders = ['train', 'val']
    stride = 35
    window_size = [(100,85),(200,100),(250,212)]
    
    box_labels = pd.read_csv('label_train.txt', header=None, delimiter=' ', 
                     names=['k', 'i', 'j', 'h', 'w'])
                     
    part_create_train_dataset = partial(create_img_dataset,stride=stride, window_size=window_size , 
                                  box_labels=box_labels, folder_size=(folders[0],32))

    part_create_val_dataset = partial(create_img_dataset,stride=stride, window_size=window_size , 
                                  box_labels=box_labels, folder_size=(folders[1],32))


    train_images = glob.glob('train\*')
    df_images = pd.DataFrame(train_images, columns=['path'])
    df_images['w'] = df_images.path.apply(lambda x: Image.open(x).size[0])
    df_images['h'] = df_images.path.apply(lambda x: Image.open(x).size[1])
    df_images['k'] = [x for x in range(1, 1001)]
    
    ret = []
    
    with Pool() as p:
        for m in tqdm(p.imap_unordered(part_create_train_dataset, df_images.iloc[:800].iterrows())):
            ret.append(m)

    ret = []

    with Pool() as p:
        for m in tqdm(p.imap_unordered(part_create_val_dataset, df_images.iloc[800:].iterrows())):
            ret.append(m)