import torch
from torch.nn import functional as F


def row_iou(pred, truth, valid, width=800, radius=15):
    distance = (pred-truth).abs()*(width-1)

    return (((2*radius-distance)/(2*radius+distance))*valid).sum(-1)/valid.sum(-1).clamp_min(1)


@torch.no_grad()
def assign(pred, target, width=800, height=320):
    p, g = len(pred['xs']), len(target['xs'])
    if not g:
        empty = torch.empty(0,dtype=torch.long,device=pred['xs'].device)
        return empty, empty
    valid = target['valid'][None]
    dis = ((pred['xs'][:,None]-target['xs'][None]).abs()*valid).sum(-1)/valid.sum(-1).clamp_min(1)*(width-1)
    scale = pred['geometry'].new_tensor([width-1,height-1])
    xy = torch.cdist(pred['geometry'][:,:2]*scale,target['geometry'][:,:2]*scale)
    angle = (pred['geometry'][:,None,2]-target['geometry'][None,:,2]).abs()
    prob = pred['logits'].softmax(-1)[:,1].clamp(1e-6,1-1e-6)
    focal = -.25*(1-prob)**2*prob.log() + .75*prob**2*(1-prob).log()
    cost = 3*(dis*xy*angle)**2 + focal[:,None]
    quality = row_iou(pred['xs'][:,None],target['xs'][None],valid,width).clamp(0,1)
    dynamic_k = quality.topk(min(10,p),dim=0).values.sum(0).long().clamp_min(1)
    matches = torch.zeros_like(cost,dtype=torch.bool)
    for j in range(g):
        matches[cost[:,j].topk(int(dynamic_k[j]),largest=False).indices,j] = True
    conflicts = matches.sum(1)>1
    if conflicts.any():
        chosen = cost[conflicts].masked_fill(~matches[conflicts],float('inf')).argmin(1)
        matches[conflicts] = False
        matches[conflicts,chosen] = True
    return matches.nonzero(as_tuple=True)


def structural_loss(xs, valid, H):
    if len(xs)<2:
        return xs.sum()*0
    y = torch.linspace(0,1,xs.shape[-1],device=xs.device).expand_as(xs)
    points = torch.stack((xs,y,torch.ones_like(xs)),-1) @ H.T
    points = points[...,:2]/points[...,2:].clamp_min(1e-4)

    weight = valid.float()
    mean = (points*weight[...,None]).sum(1)/weight.sum(1,keepdim=True).clamp_min(1)
    centered = points-mean[:,None]
    slope = (centered[...,0]*centered[...,1]*weight).sum(1)/((centered[...,1]**2*weight).sum(1)+1e-6)
    normals = F.normalize(torch.stack((torch.ones_like(slope),-slope),-1),dim=-1)
    cross = normals[:,0,None]*normals[None,:,1]-normals[:,1,None]*normals[None,:,0]
    offdiag = ~torch.eye(len(xs),dtype=torch.bool,device=xs.device)
    return F.smooth_l1_loss(cross[offdiag],torch.zeros_like(cross[offdiag]))


def criterion(output, targets, width=800):
    zero = output['stages'][0]['xs'].sum()*0
    terms = {k:zero for k in ('cls','reg','riou','structure','seg')}
    for stage_index, stage in enumerate(output['stages']):
        for b,target in enumerate(targets):
            pred = {k:v[b] for k,v in stage.items()}
            pi, gi = assign(pred,target,width)
            labels = torch.zeros(len(pred['xs']),dtype=torch.long,device=zero.device)
            labels[pi] = 1
            ce = F.cross_entropy(pred['logits'],labels,reduction='none')
            alpha = torch.where(labels.bool(),.25,.75)
            terms['cls'] = terms['cls']+(alpha*(1-(-ce).exp())**2*ce).sum()/max(len(pi),1)
            if len(pi):
                pr = torch.cat((pred['geometry'][pi],pred['length'][pi,None]),-1)
                gt = torch.cat((target['geometry'][gi],target['length'][gi,None]),-1)
                terms['reg'] = terms['reg']+F.smooth_l1_loss(pr,gt)
                terms['riou'] = terms['riou']+(1-row_iou(pred['xs'][pi],target['xs'][gi],target['valid'][gi],width)).mean()
                if stage_index==len(output['stages'])-1:
                    terms['structure'] = terms['structure']+structural_loss(pred['xs'][pi],target['valid'][gi],output['H'][b])
    for k in ('cls','reg','riou','structure'):
        terms[k] = terms[k]/len(targets)
    terms['seg'] = F.binary_cross_entropy_with_logits(output['seg'],torch.stack([t['seg'] for t in targets]))
    total = sum(terms[k]*v for k,v in zip(terms,(6,.5,2,2,1)))
    return total, {k:float(v.detach()) for k,v in terms.items()}
