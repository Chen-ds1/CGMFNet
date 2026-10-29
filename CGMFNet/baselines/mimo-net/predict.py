from datetime import datetime
import config as C
from inference import Predictor,save_prediction
from evaluate import write_csv

def main():
    files=sorted(C.PREDICT_RGB.glob('*.png'))
    if not files:raise ValueError('No RGB PNG files')
    runner=Predictor(C.CHECKPOINT)
    runner.warmup(files[0],C.PREDICT_DEPTH/files[0].name)
    out=C.RUN_ROOT/('predictions_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    rows=[]
    for f in files:
        im,p,l,r,t=runner.predict(f,C.PREDICT_DEPTH/f.name)
        save_prediction(out,f.stem,im,p,l,r);rows.append(dict(sample_id=f.stem,**t))
    write_csv(out/'per_image_timing.csv',rows)
    print(out)

if __name__=='__main__':main()
