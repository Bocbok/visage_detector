from model import get_model
import glob
import pandas as pd
from PIL import Image
from visage_detector import visage_detector

# Prediction threshold
pred_t = 0.980422

model = get_model()
model.load_weights('model_weights/final04.47-0.08.hdf5')

box_labels = pd.read_csv('label_train.txt', header=None, delimiter=' ', 
                     names=['k', 'i', 'j', 'h', 'w'])

train_images = glob.glob('train\*')
df_images = pd.DataFrame(train_images, columns=['path'])
df_images['w'] = df_images.path.apply(lambda x: Image.open(x).size[0])
df_images['h'] = df_images.path.apply(lambda x: Image.open(x).size[1])
df_images['k'] = [x for x in range(1, 1001)]

vd = visage_detector(model, stride=30, ws=[(100,85),(200,100),(250,212)] , pred_t=pred_t)

test_images = glob.glob('test\*')
test_images = pd.DataFrame(test_images, columns=['path'])
test_images['w'] = test_images.path.apply(lambda x: Image.open(x).size[0])
test_images['h'] = test_images.path.apply(lambda x: Image.open(x).size[1])
test_images['k'] = [x for x in range(1, 501)]

preds = vd.predict(test_images)
preds = vd.get_merged_box_score(test_images, preds)
preds = preds[preds.score > pred_t]

preds['score'] = preds.apply(lambda x: f"{x.score:0.2f}", axis=1)

preds.to_csv('detection.txt', index=False, columns=['k', 'i', 'j', 'h', 'w', 'score'], header=False, sep=' ')