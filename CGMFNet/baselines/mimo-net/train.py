import json,random,time
import numpy as np
import torch
from torch.utils.data import DataLoader
import config as C
from data import audit,validate_splits,RiceDataset
from model import MIMONet
from losses import multitask_loss

def model_config():
    return dict(backbone=C.BACKBONE,width=C.WIDTH,image_hw=C.IMAGE_HW,depth_mode=C.DEPTH_MODE)

def seed_everything():
    random.seed(C.SEED);np.random.seed(C.SEED);torch.manual_seed(C.SEED)
    torch.cuda.manual_seed_all(C.SEED)

def main():
    seed_everything(); ids,_=audit(C.DATA_ROOT)
    if not C.SPLITS.exists(): raise FileNotFoundError('Run prepare_data.py first')
    splits=json.loads(C.SPLITS.read_text(encoding='utf-8'));validate_splits(splits,ids)
    training_checkpoint=C.RUN_ROOT/'best.pt'
    if training_checkpoint.exists(): raise FileExistsError('Existing best.pt: change RUN_ROOT for a new experiment; no overwrite')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg=model_config(); model=MIMONet(**cfg,pretrained=C.PRETRAINED).to(device)
    loaders={}
    for part in ['train','val']:
        ds=RiceDataset(C.DATA_ROOT,splits[part],C.IMAGE_HW,C.LINE_WIDTH,C.DEPTH_MODE,C.DEPTH_SCALE_METERS,part=='train')
        loaders[part]=DataLoader(ds,batch_size=C.BATCH_SIZE,shuffle=part=='train',num_workers=C.NUM_WORKERS)
    optimizer=torch.optim.AdamW(model.parameters(),lr=C.LR,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,C.EPOCHS)
    C.RUN_ROOT.mkdir(parents=True,exist_ok=True)
    metadata={'implementation':'paper-informed rice adaptation','model':cfg,'splits':splits,
        'seed':C.SEED,'epochs':C.EPOCHS,'batch_size':C.BATCH_SIZE,'learning_rate':C.LR,
        'optimizer':'AdamW','pretrained':C.PRETRAINED,'line_width':C.LINE_WIDTH,
        'early_stop_patience':getattr(C,'EARLY_STOP_PATIENCE',15),
        'depth_scale_meters':C.DEPTH_SCALE_METERS,'torch':torch.__version__,
        'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None}
    (C.RUN_ROOT/'experiment.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    best=float('inf'); history=[];stale=0
    for epoch in range(C.EPOCHS):
        record={'epoch':epoch+1}; start=time.perf_counter()
        for part,loader in loaders.items():
            model.train(part=='train'); total=0.; count=0
            with torch.set_grad_enabled(part=='train'):
                for rgb,depth,plant,line,_ in loader:
                    rgb,depth,plant,line=[a.to(device) for a in [rgb,depth,plant,line]]
                    if part=='train': optimizer.zero_grad(set_to_none=True)
                    loss=multitask_loss(model(rgb,depth),plant,line)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                    if part=='train':
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                    total+=float(loss.detach())*len(rgb);count+=len(rgb)
            record[part+'_loss']=total/count
        scheduler.step(); record['seconds']=time.perf_counter()-start;history.append(record)
        state={'model':model.state_dict(),'model_config':cfg,'metadata':metadata,'epoch':epoch+1,
               'val_loss':record['val_loss'],'optimizer':optimizer.state_dict()}
        torch.save(state,C.RUN_ROOT/'last.pt')
        if record['val_loss']<best:
            best=record['val_loss'];torch.save(state,training_checkpoint);stale=0
        else:stale+=1
        (C.RUN_ROOT/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(record,flush=True)
        if stale>=getattr(C,'EARLY_STOP_PATIENCE',15):
            print('Early stopping: validation loss did not improve.',flush=True);break
    print('Training complete. Run evaluate.py on the held-out test split.')

if __name__=='__main__':main()
