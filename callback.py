from tensorflow.keras.callbacks import Callback


# Keras Callback computing f1, precision, recall on epoch end

class f1_class(Callback):
    def __init__(self, val_generator=None):
        super(Callback, self).__init__()                
        self.validation_generator = val_generator

            
    def on_train_begin(self, logs={}):
        self.f1s = []
        self.f2s = []
        self.ps = []
        self.re = []
        self.y_true = np.array(self.validation_generator.labels)
        
        
    def get_f1_thresh(self, y_true, y_pred):
        df = pd.DataFrame(y_true, columns=['y_true'])
        df['y_pred'] = y_pred
        cleaner = df.duplicated('y_pred', keep='last')
        df = df.sort_values('y_pred', ascending=False)
        df['vp'] = df.y_true.cumsum()
        df['fp'] = (df.y_true == 0).cumsum()
        df['fn'] = sum(df.y_true) - df['vp']
        df['prec'] = df.vp / (df.vp + df.fp)
        df['rec'] = df.vp / (df.vp + df.fn)
        df = df.loc[~cleaner]
        df['f1'] = 2* (df.prec * df.rec) / (df.prec + df.rec)
        df['f2'] = 5* (df.prec * df.rec) / (4*df.prec + df.rec)
        bestf1 = df['f1'].idxmax()
        bestf2 = df['f2'].idxmax()
        return df.loc[bestf1], df.loc[bestf2].f2
        
        
    def on_epoch_end(self, epoch, logs=None):   
        if epoch>=10:
            y_pred = self.model.predict_generator(self.validation_generator).flatten()


            best, f2 = self.get_f1_thresh(self.y_true, y_pred)

            score = best.f1
            p = best.prec
            r = best.rec

            self.f1s.append(score)
            self.ps.append(p)
            self.re.append(r)
            self.f2s.append(f2)
            
            logs['f1'] = score

            print("interval evaluation - epoch: {:d} - f1: {:.6f}, f2: {:.6f}, precision: {:.6f}, recall: {:.6f}".format(epoch, score, f2, p, r))
        else:
            logs['f1'] = 0