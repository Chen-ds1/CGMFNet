import torch
from torch.nn import functional as F

def lovasz_hinge(logits,target):
    losses=[]
    for z,y in zip(logits,target):
        z,y=z.flatten(),y.flatten()
        errors=1-z*(2*y-1)
        errors,perm=torch.sort(errors,descending=True)
        gt=y[perm]; total=gt.sum()
        intersection=total-gt.cumsum(0)
        union=total+(1-gt).cumsum(0)
        grad=1-intersection/union.clamp_min(1e-6)
        if len(grad)>1: grad=torch.cat([grad[:1],grad[1:]-grad[:-1]])
        losses.append(torch.dot(F.relu(errors),grad))
    return torch.stack(losses).mean()

def multitask_loss(out,plant,line):
    z=out['plant'].float(); y=plant.float(); p=z.sigmoid()
    pt=p*y+(1-p)*(1-y)
    alpha=0.25*y+0.75*(1-y)
    seg=(alpha*(1-pt).pow(2)*F.binary_cross_entropy_with_logits(z,y,reduction='none')).mean()
    z=out['line'].float(); y=line.float(); p=z.sigmoid()
    dims=(1,2,3); tp=(p*y).sum(dims); fp=(p*(1-y)).sum(dims); fn=((1-p)*y).sum(dims)
    ft=(1-(tp+1)/(tp+0.7*fn+0.3*fp+1)).pow(0.75).mean()
    dice=(1-(2*tp+1)/(p.sum(dims)+y.sum(dims)+1)).mean()
    combo=0.5*F.binary_cross_entropy_with_logits(z,y)+0.5*dice
    line_loss=ft+combo+lovasz_hinge(z,y)
    return seg+line_loss
