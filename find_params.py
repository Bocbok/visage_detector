from util import *
from functools import partial
from tqdm import tqdm
import glob

if __name__ == '__main__':
    box_labels = pd.read_csv('label_train.txt', header=None, delimiter=' ', 
                     names=['k', 'i', 'j', 'h', 'w'])

    train_images = glob.glob('train\*')
    df_images = pd.DataFrame(train_images, columns=['path'])
    df_images['w'] = df_images.path.apply(lambda x: Image.open(x).size[0])
    df_images['h'] = df_images.path.apply(lambda x: Image.open(x).size[1])
    df_images['k'] = [x for x in range(1, 1001)]

    f_part = partial(get_params_perf, imgs=df_images, labels=box_labels)


    strides = [30,35,40]
    sizes = [[(100,85),(200,100),(250,212)]]

    for x in range(75, 400, 25):
        sizes.append([(x, int(x*0.50))])
        sizes.append([(x, int(x*0.65))])
        sizes.append([(x, int(x*0.85))])
        sizes.append([(x, x)])
    
    paramlist = list(itertools.product(strides, sizes))

    print(paramlist, len(paramlist))

    ret = []
    
    with Pool() as p:
        for m in tqdm(p.imap_unordered(f_part, paramlist)):
            ret.append(m)

    vp, box_list, stats = zip(*ret)
    stats = pd.DataFrame(stats)
    stats['mean/num_im'] = stats['mean']*100 / stats['num_im']
    stats.sort_values('acc', ascending=False)