"""Exact-endpoint circle/straight/circle and three-circle connections.

Common tangents and equal-radius circle intersections supply forward Dubins
families. Free final heading is sampled; mixed forward/reverse words are not
enumerated. Collision and motion limits are checked by each consumer.
"""
import numpy as np
from navigation.kinematics import turning_residual
from .motion import twists
from .footprint import wrap,compact_poses


def motion_valid(poses,policy):
    rates=twists(np.asarray(poses),np.ones(len(poses)-1))
    return bool(np.max(abs(rates[:,1]),initial=0.)<=1e-6 and
        np.min(turning_residual(rates[:,[0,2]],policy),initial=0.)>=-1e-7 and
        (policy['allow_reverse'] or np.min(rates[:,0],initial=0.)>=-1e-7))


def connections(start,goal,radius,step=.15,*,reverse=False):
    """Yield exact-endpoint arc paths in increasing length, without pose snapping."""
    start=np.asarray(start,float);goal=np.asarray(goal,float)
    offset=np.pi if reverse else 0.;a=start.copy();a[2]+=offset
    headings=[goal[2]+offset] if len(goal)==3 else list(dict.fromkeys([
        np.arctan2(*(goal[:2]-start[:2])[::-1]),a[2],*np.arange(16)*np.pi/8]))
    candidates=[]
    def mod(x):
        x=x%(2*np.pi)
        return 0. if min(x,2*np.pi-x)<1e-8 else x
    for heading in headings:
        for first in (-1.,1.):
            c1=a[:2]+first*radius*np.array([-np.sin(a[2]),np.cos(a[2])])
            for last in (-1.,1.):
                c2=goal[:2]+last*radius*np.array([-np.sin(heading),np.cos(heading)])
                delta=c2-c1;distance=np.linalg.norm(delta)
                if distance<1e-8:
                    if first!=last:continue
                    segments=[(first/radius,radius*mod(first*(heading-a[2])))]
                else:
                    sine=(last-first)*radius/distance
                    if abs(sine)>1.+1e-10:continue
                    tangent=np.arctan2(delta[1],delta[0])-np.arcsin(np.clip(sine,-1.,1.))
                    segments=[(first/radius,radius*mod(first*(tangent-a[2]))),
                        (0.,np.sqrt(max(0.,distance**2-((last-first)*radius)**2))),
                        (last/radius,radius*mod(last*(heading-tangent)))]
                candidates.append((sum(length for _,length in segments),segments,heading))
        for sign in (-1.,1.):
            c1=a[:2]+sign*radius*np.array([-np.sin(a[2]),np.cos(a[2])])
            c3=goal[:2]+sign*radius*np.array([-np.sin(heading),np.cos(heading)])
            delta=c3-c1;distance=np.linalg.norm(delta)
            if not 1e-8<distance<=4*radius:continue
            normal=np.array([-delta[1],delta[0]])/distance
            height=np.sqrt(max(0.,4*radius**2-distance**2/4))
            for side in (-1.,1.):
                c2=(c1+c3)/2+side*normal*height
                p12=(c1+c2)/2;p23=(c2+c3)/2
                yaw12=np.arctan2(*(p12-c1)[::-1])+sign*np.pi/2
                yaw23=np.arctan2(*(p23-c2)[::-1])-sign*np.pi/2
                segments=[(sign/radius,radius*mod(sign*(yaw12-a[2]))),
                    (-sign/radius,radius*mod(-sign*(yaw23-yaw12))),
                    (sign/radius,radius*mod(sign*(heading-yaw23)))]
                candidates.append((sum(length for _,length in segments),segments,heading))
    for length,segments,heading in sorted(candidates,key=lambda item:item[0]):
        pose=a.copy();pieces=[pose[None,:].copy()]
        for curvature,distance in segments:
            if distance<1e-9:continue
            n=max(1,int(np.ceil(distance/step)))
            travel=np.arange(1,n+1)*distance/n;angle=curvature*travel
            chord=travel*np.sinc(angle/(2*np.pi));yaw=pose[2]+angle/2
            piece=pose+np.c_[chord*np.cos(yaw),chord*np.sin(yaw),angle]
            pieces.append(piece);pose=piece[-1]
        poses=np.vstack(pieces);poses[:,2]-=offset
        if len(poses)==1:poses=np.vstack([poses,poses])
        if np.linalg.norm(poses[-1,:2]-goal[:2])>1e-6 or abs(wrap(poses[-1,2]-heading+offset))>1e-6:
            raise RuntimeError('Invalid constant-curvature endpoint')
        poses[-1,:2]=goal[:2]
        yield poses,float(length)


def reference_tails(reference,start,policy):
    """Retain nearby geometric guidance; consumers must certify actual-start motion."""
    reference=np.asarray(reference);start=np.asarray(start)
    delta=np.diff(reference[:,:2],axis=0);length=np.linalg.norm(delta,axis=1)
    if not len(length):return
    fraction=np.clip(np.sum((start[:2]-reference[:-1,:2])*delta,axis=1)/np.maximum(length**2,1e-15),0,1)
    distance=np.linalg.norm(start[:2]-reference[:-1,:2]-fraction[:,None]*delta,axis=1)
    yaws=reference[:-1,2]+fraction*wrap(np.diff(reference[:,2]))
    angles=abs(wrap(yaws-start[2]));i=int(np.argmin(distance+.05*angles))
    if distance[i]>.25:return
    yaw=yaws[i]
    if abs(wrap(yaw-start[2]))>.4:return
    # This tail starts on the old route, not at the actual robot pose. Keep that
    # distinction explicit in the planner result; local trajectory generation
    # still starts at the current observed pose and checks all motion limits.
    yield reference[i:]
    turns=wrap(np.diff(reference[:,2]));mid=reference[:-1,2]+turns/2
    direction=np.where(length>1e-7,np.where(np.sum(delta*np.c_[np.cos(mid),np.sin(mid)],axis=1)<0,-1,1),
                       np.where(turns<0,-2,2))
    end=i+1
    while end<len(direction) and direction[end]==direction[i]:end+=1
    # Do not shortcut a later direction change or a planned turn.
    if abs(direction[i])==2 and np.linalg.norm(reference[i,:2]-start[:2])<1e-6:
        done=wrap(start[2]-reference[i,2]);total=turns[i]
        if done*total>=-1e-10 and abs(done)<=abs(total)+1e-6:
            candidate=compact_poses(np.vstack([start,reference[i+1:]]))
            if len(candidate)>=2 and motion_valid(candidate,policy):yield candidate
    arc=np.r_[0.,np.cumsum(length)];progress=arc[i]+fraction[i]*length[i]
    indices=sorted(set(min(end,int(np.searchsorted(arc,progress+d))) for d in (.2,.5,.8,1.5,2.5)))
    for index in indices:
        # A connector must keep the current section's travel direction. It
        # still passes exact-start kinematic and complete swept-grid checks.
        directions=(True,) if direction[i]==-1 else (False,) if direction[i]==1 else ()
        for reverse in directions:
            if reverse and not policy['allow_reverse']:continue
            for path,total in connections(start,reference[index],policy['min_turning_radius_m'],.1,reverse=reverse):
                if total>1.6*max(.1,arc[index]-progress)+.2:break
                candidate=np.vstack([path,reference[index+1:]])
                if motion_valid(candidate,policy):yield compact_poses(candidate)
