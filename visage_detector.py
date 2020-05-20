from PIL import Image
import numpy as np
import pandas as pd
from util import *



# custom IoU func for more convenience
def local_IoU(box1, box2):
    
    l_intersection = min(box1.j + box1.w, box2.j + box2.w) - max(box1.j, box2.j)
    h_intersection = min(box1.i + box1.h, box2.i + box2.h) - max(box1.i, box2.i) 
    # No overlap
    if l_intersection <= 0 or h_intersection <=0:
        return 0
    
    I = l_intersection * h_intersection
    
    U = box1.w * box1.h + box2.w * box2.h - I
    
    return I / U


class visage_detector():
    
    def __init__(self, model, pred_t, stride=35, ws=[(100,85),(200,100),(250,212)]):
        self.model = model
        self.s = stride
        self.ws = ws
        self.pred_t = pred_t
        
        
    def get_box_groups(self, boxes, thresh=0.2):
        ret = boxes.copy()
        ret['area'] = boxes.w * boxes.h
        ret = ret.sort_values('area')

        groups = [[ret.iloc[0].name]]

        for idx, row in ret.iloc[1:].iterrows():
            create_new_group = True
            for g in groups:
                if (ret.loc[g].apply(lambda x: local_IoU(x.drop(['score','area']), row.drop(['score','area'])), axis=1)).max() > thresh:
                    g.append(idx)
                    create_new_group = False

            if create_new_group:
                groups.append([idx])
                
        return groups
    
    
    def merge_boxes(self, boxes_groups, boxes):
        merged_boxes = []
        for g in boxes_groups:
            b = boxes.loc[g]       
            b = b.mean().astype(int)
            merged_boxes.append(b)

        return pd.DataFrame(merged_boxes)


    def get_merged_box_score(self, imgs, merged_boxes):
        
        to_pred = []
        
        merged_boxes.apply(lambda x: to_pred.append(np.asarray(Image.open(imgs.loc[x.k-1, 'path'])
                                                                          .crop((x.j, x.i, x.j+x.w, x.i+x.h))
                                                                          .resize((32, 32)))*1./255), axis=1)
        
        print(len(to_pred), to_pred[0].shape)
        
        merged_boxes['score'] = self.model.predict(np.array(to_pred)).flatten()
        
        return merged_boxes
        
    
    
    def predict_one_image(self, img, merge=True):
        boxes = sliding_window(img, self.s, self.ws)
        
        to_pred = []

        with Image.open(img.path) as im:
            boxes.apply(lambda x: to_pred.append(
                np.asarray(im.crop((x.j, x.i, x.j+x.w, x.i+x.h))
                           .resize((32, 32)))*1./255), axis=1)
        
        to_pred = np.array(to_pred)
        
        boxes['score'] = self.model.predict(to_pred).flatten()
        best_boxes = boxes[boxes.score > self.pred_t]
        
        if len(best_boxes)==0:
            return None
        
        if merge == False:
            return best_boxes
        else:
            merged_boxes = self.merge_boxes(self.get_box_groups(best_boxes), best_boxes).drop(columns='score')

            to_del = []

            if len(merged_boxes) > 1:
                for pair in combinations(range(len(merged_boxes)), r=2):
                    b1 = merged_boxes.iloc[pair[0]]
                    b2 = merged_boxes.iloc[pair[1]]

                    I = local_IoU(b1, b2)
                    if I > 0.4:
                        to_del.append(merged_boxes.iloc[pair[1]].name)

            return merged_boxes.drop(np.unique(to_del))
    
    
    def predict(self, df_imgs):
        predictions = []
        
        for idx, row in df_imgs.iterrows():
            predictions.append(self.predict_one_image(row))
            
            
        return pd.concat(predictions).reset_index().drop(columns='index')
     
        
        
    def evaluate(self, X, y): 
        preds = self.predict(X)
        preds = self.get_merged_box_score(X, preds)
        preds = preds[preds.score > self.pred_t].drop(columns='score')
        
        tp = 0
        fp = 0
        fn = 0
        
        y_test = y.copy()
        y_test['counted'] = 0
        
        for idx, box in preds.iterrows():
            y_test = y_test[y_test.counted == 0]
            ious = y_test.drop(columns='counted').apply(lambda x: local_IoU(x, box), axis=1)
            if len(ious) != 0:
                if ious.max() >= 0.5:
                    tp += 1
                    y_test.loc[ious.idxmax(), 'counted'] = 1
                else:
                    print(box.k)
                    fp += 1
                     
            
        fn = len(y) - tp
        
        p = tp / (tp + fp) if (tp+fp)!=0 else 0
        r = tp / (tp+fn) if (tp+fn)!=0 else 0
        
        f1 = 2*(p*r)/(p+r) if (p+r)!=0 else 0
        
        print(f'TP: {tp}, FP: {fp}, FN: {fn}')
        print(f'f1: {f1}, precision: {p}, recall: {r}')
        
        
        return f1, p, r
        
        
    def crop_resize(self, box, path):
        with Image.open(path) as im:
            return np.asarray(im.crop((box.j, box.i, box.j+box.w, box.i+box.h)).resize((32, 32)))*1./255
