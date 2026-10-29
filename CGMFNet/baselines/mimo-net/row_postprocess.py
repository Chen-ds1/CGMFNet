import numpy as np
from scipy.ndimage import gaussian_filter1d, binary_closing

DEFAULTS = dict(threshold=0.5, min_support=0.15, radius_fraction=0.015,
                max_slope=1.5, slope_steps=121, scan_step=4)

def remove_crossing_candidates(lines, raw, quality):
    kept=[]
    for i in sorted(range(len(lines)),key=lambda j:quality[j],reverse=True):
        x,y=lines[i][:,0],lines[i][:,1]
        a,b=np.polyfit(y,x,1)
        conflict=False
        for j in kept:
            old=lines[j];ao,bo=np.polyfit(old[:,1],old[:,0],1)
            lo=max(y.min(),old[:,1].min());hi=min(y.max(),old[:,1].max())
            if hi<=lo:continue
            d0=(a-ao)*lo+b-bo;d1=(a-ao)*hi+b-bo
            if d0*d1<=0:
                conflict=True;break
        if not conflict:kept.append(i)
    kept.sort(key=lambda j:np.mean(lines[j][:,0]))
    return [lines[i] for i in kept],[raw[i] for i in kept]

def band_centres(prob, threshold, scan_step=4):
    h,w=prob.shape; points=[]
    for y in range(1,h-1,scan_step):
        signal=gaussian_filter1d(prob[y-1:y+2].mean(0),1)
        mask=binary_closing(signal>=threshold,structure=np.ones(3))
        edges=np.diff(np.r_[False,mask,False].astype(int))
        for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            if end-start<2:continue
            xs=np.arange(start,end);weights=signal[start:end]
            points.append([float(np.average(xs,weights=weights)),float(y),float(weights.mean())])
    return np.asarray(points,float).reshape(-1,3)

def extract_rows(prob, threshold=0.5, min_support=0.15, radius_fraction=0.015,
                 max_slope=1.5, slope_steps=121, scan_step=4):
    h,w=prob.shape
    points=band_centres(prob,threshold,scan_step)
    if len(points)<3:return [],[]
    available=np.ones(len(points),bool); yref=(h-1)/2
    radius=max(3.,w*radius_fraction); bin_width=max(2.,radius/2)
    slopes=np.linspace(-max_slope,max_slope,slope_steps)
    xmin=-max_slope*h-w; xmax=max_slope*h+2*w
    bins=int(np.ceil((xmax-xmin)/bin_width))+1

    intercepts=points[None,:,0]-slopes[:,None]*(points[None,:,1]-yref)
    indices=np.clip(np.rint((intercepts-xmin)/bin_width).astype(int),0,bins-1)
    required=max(5,int(np.ceil((h-2)/scan_step*min_support)))
    lines=[];raw=[];quality=[]
    while available.sum()>=required:
        votes=np.zeros((len(slopes),bins))
        for i in range(len(slopes)):
            votes[i]=np.bincount(indices[i,available],weights=points[available,2],minlength=bins)
        votes=gaussian_filter1d(votes,1,axis=1)
        si,bi=np.unravel_index(votes.argmax(),votes.shape)
        a=slopes[si];b=xmin+bi*bin_width
        chosen=np.zeros(len(points),bool)

        for _ in range(4):
            residual=np.abs(points[:,0]-(a*(points[:,1]-yref)+b))
            candidates=np.flatnonzero(available&(residual<=radius))
            chosen[:]=False
            for y in np.unique(points[candidates,1]):
                ids=candidates[points[candidates,1]==y]
                chosen[ids[np.argmin(residual[ids])]]=True
            if chosen.sum()<3:break
            p=points[chosen]; x=p[:,0]; yc=p[:,1]-yref
            error=np.abs(x-(a*yc+b))
            robust=np.minimum(1.,(radius/2)/np.maximum(error,1e-6))
            a,b=np.polyfit(yc,x,1,w=np.sqrt(p[:,2]*robust))
        if not chosen.any():

            dist=np.abs(points[:,0]-(a*(points[:,1]-yref)+b));dist[~available]=np.inf
            available[dist.argmin()]=False
            continue

        residual=np.abs(points[:,0]-(a*(points[:,1]-yref)+b))
        available[available&(residual<=radius)]=False
        p=points[chosen]
        if len(p)<required or np.ptp(p[:,1])<h*min_support:continue
        yy=np.array([p[:,1].min(),p[:,1].max()]);xx=a*(yy-yref)+b

        line=np.c_[xx,yy]
        score=float(len(p)*p[:,2].mean())
        duplicate=False
        for j,old in enumerate(lines):
            ao,bo=np.polyfit(old[:,1]-yref,old[:,0],1)
            sep=np.abs((a-ao)*(np.array([0,h-1])-yref)+(b-bo))
            if np.max(sep)<2*radius:
                duplicate=True
                if score>quality[j]:lines[j]=line;raw[j]=p[:,:2];quality[j]=score
                break
        if not duplicate:lines.append(line);raw.append(p[:,:2]);quality.append(score)
    return remove_crossing_candidates(lines,raw,quality)
