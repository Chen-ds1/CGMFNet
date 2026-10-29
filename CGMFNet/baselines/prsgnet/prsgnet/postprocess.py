import cv2
import numpy as np


def decode(stage, height, width, score=.5, nms_distance=15, max_rows=20):
    prob = stage['logits'].softmax(-1)[:,1]
    xs = stage['xs'].detach().cpu().numpy()*(width-1)
    geom = stage['geometry'].detach().cpu().numpy()
    lengths = stage['length'].detach().cpu().numpy()
    n = xs.shape[1]
    yy = np.linspace(height-1,0,n)
    result = []
    for idx in prob.argsort(descending=True).tolist():
        if float(prob[idx])<score:
            break
        start = int(np.clip(round(geom[idx,1]*(n-1)),0,n-1))
        end = min(n,start+max(0,round(lengths[idx]*n)))
        valid = (np.arange(n)>=start)&(np.arange(n)<end)&(xs[idx]>=0)&(xs[idx]<=width-1)
        if valid.sum()<2:
            continue
        duplicate = False
        for old in result:
            overlap = valid & old['valid']
            if overlap.sum()>=2 and np.abs(xs[idx,overlap]-old['xs'][overlap]).mean()<nms_distance:
                duplicate = True
                break
        if not duplicate:
            result.append({'score':float(prob[idx]),'points':np.c_[xs[idx,valid],yy[valid]].tolist(),
                           'xs':xs[idx], 'valid':valid})
        if len(result)>=max_rows:
            break
    return result


def curve_mask(points,height,width,thickness=30):
    mask = np.zeros((height,width),np.uint8)
    if len(points)>1:
        cv2.polylines(mask,[np.rint(points).astype(np.int32)],False,1,thickness)
    return mask.astype(bool)


def counts(predicted, truth, height, width, threshold):
    pm = [curve_mask(p,height,width) for p in predicted]
    gm = [curve_mask(g,height,width) for g in truth]
    edges = []
    for p in pm:
        edges.append([j for j,g in enumerate(gm) if np.logical_and(p,g).sum()/max(1,np.logical_or(p,g).sum())>=threshold])
    matched = {}
    def augment(i,seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or augment(matched[j],seen):
                matched[j]=i
                return True
        return False
    tp = sum(augment(i,set()) for i in range(len(pm)))
    return np.array([tp,len(pm)-tp,len(gm)-tp])
