"""Project DWA candidate positions onto the geometric reference."""
import numpy as np


def segments(reference):
    delta=np.diff(reference[:,:2],axis=0);length=np.linalg.norm(delta,axis=1);keep=length>1e-7
    return reference[:-1,:2][keep],delta[keep],length[keep],np.r_[0.,np.cumsum(length[keep])]


def project_progress(xy,reference,prepared=None):
    a,delta,length,arc=segments(reference) if prepared is None else prepared
    if not len(length):return None
    fraction=np.clip(np.sum((xy[:,None]-a)*delta,axis=2)/length**2,0,1)
    distance=np.linalg.norm(xy[:,None]-a-fraction[:,:,None]*delta,axis=2)
    index=np.argmin(distance,axis=1);rows=np.arange(len(xy))
    return arc[index]+fraction[rows,index]*length[index],distance[rows,index],float(arc[-1])
