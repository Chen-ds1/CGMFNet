import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def curve(p):
    p=np.asarray(p,float)
    order=np.argsort(p[:,1]); p=p[order]
    y=np.unique(p[:,1]); x=np.array([p[p[:,1]==v,0].mean() for v in y])
    return x,y

def angle(p):
    p=np.asarray(p,float)
    _,_,v=np.linalg.svd(p-p.mean(0),full_matrices=False)
    return float(np.degrees(np.arctan2(v[0,1],v[0,0]))%180)

def angle_error(a,b):
    d=abs(angle(a)-angle(b))%180
    return min(d,180-d)

def comparison(a,b,min_overlap=0.5):
    xa,ya=curve(a); xb,yb=curve(b)
    lo=max(ya.min(),yb.min()); hi=min(ya.max(),yb.max())
    if hi<=lo or (hi-lo)/max(np.ptp(ya),1)<min_overlap: return None
    ys=np.linspace(lo,hi,100)
    return {'lateral_px':float(np.abs(np.interp(ys,ya,xa)-np.interp(ys,yb,xb)).mean()),
            'angle_deg':angle_error(a,b),
            'vertical_coverage':float((hi-lo)/max(np.ptp(ya),1))}

def transform(p,h):
    q=np.c_[p,np.ones(len(p))]@np.asarray(h,float).T
    if np.any(np.abs(q[:,2])<1e-8): raise ValueError('Invalid homography projection')
    return q[:,:2]/q[:,2,None]

def ground_error(a,b,h):
    a,b=transform(a,h),transform(b,h)

    def distance(points,line):
        _,_,v=np.linalg.svd(line-line.mean(0),full_matrices=False)
        n=np.array([-v[0,1],v[0,0]])
        return np.abs((points-line.mean(0))@n).mean()
    return float((distance(a,b)+distance(b,a))/2),angle_error(a,b)

def match_lines(gt,pred,max_px=30,max_angle=15,min_overlap=0.5,homography=None):
    n,m=len(gt),len(pred)
    cost=np.full((n,m),1e6); details={}
    for i,a in enumerate(gt):
        for j,b in enumerate(pred):
            d=comparison(a,b,min_overlap)
            if d is not None and d['lateral_px']<=max_px and d['angle_deg']<=max_angle:
                cost[i,j]=d['lateral_px']/max_px+d['angle_deg']/max_angle
                details[i,j]=d
    matches=[]
    if n and m:
        rows,cols=linear_sum_assignment(cost)
        for i,j in zip(rows,cols):
            if (i,j) in details:
                d=dict(details[i,j],gt_index=int(i),pred_index=int(j))
                if homography is not None:
                    d['lateral_cm'],d['ground_angle_deg']=ground_error(gt[i],pred[j],homography)
                matches.append(d)
    return matches,n-len(matches),m-len(matches)

def extract_lines(prob,threshold=0.5,min_span=0.15):
    h,w=prob.shape; tracks=[]; step=4
    for y in range(h-1,-1,-step):
        signal=gaussian_filter1d(prob[max(0,y-1):min(h,y+2)].mean(0),1)
        peaks,_=find_peaks(np.pad(signal,1),height=threshold,distance=max(3,int(w*0.008)))
        peaks=peaks-1; peaks=peaks[(peaks>=0)&(peaks<w)]
        active=[i for i,t in enumerate(tracks) if t[-1][1]-y<=step*5]
        used=set()
        if active and len(peaks):
            expected=[]
            for i in active:
                t=np.array(tracks[i][-8:])
                if len(t)>1:
                    fit=np.polyfit(t[:,1],t[:,0],1); expected.append(np.polyval(fit,y))
                else: expected.append(t[-1,0])
            costs=np.abs(np.array(expected)[:,None]-peaks[None,:])
            rr,cc=linear_sum_assignment(costs)
            for r,c in zip(rr,cc):
                if costs[r,c]<=max(10,w*0.025):
                    tracks[active[r]].append([float(peaks[c]),float(y)]); used.add(c)
        for i,x in enumerate(peaks):
            if i not in used: tracks.append([[float(x),float(y)]])
    result=[]; raw=[]
    for t in tracks:
        p=np.asarray(t)
        if len(p)<6 or np.ptp(p[:,1])<h*min_span: continue
        coef=np.polyfit(p[:,1],p[:,0],1)
        ys=np.array([p[:,1].min(),p[:,1].max()]); xs=np.polyval(coef,ys)
        result.append(np.c_[np.clip(xs,0,w-1),ys]); raw.append(p)
    return result,raw

def segmentation_stats(pred,target):
    p=np.asarray(pred,bool); y=np.asarray(target,bool)
    tp=int((p&y).sum()); fp=int((p&~y).sum()); fn=int((~p&y).sum()); tn=int((~p&~y).sum())
    def ratio(a,b): return float(a/b) if b else None
    fg=ratio(tp,tp+fp+fn); bg=ratio(tn,tn+fp+fn)
    return {'iou':fg,'miou':float(np.mean([v for v in [fg,bg] if v is not None])),
            'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),'dice':ratio(2*tp,2*tp+fp+fn),
            'tp_pixels':tp,'fp_pixels':fp,'fn_pixels':fn,'tn_pixels':tn}
