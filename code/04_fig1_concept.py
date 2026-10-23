"""Fig. 1: SRT concept figure from two real NHANES participants (same mean sleep, different SRT)."""
import numpy as np, pandas as pd, importlib.util, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
spec=importlib.util.spec_from_file_location('m','code/01b_compute_metrics_indiv.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import os
sel={'A':int(os.environ.get('A',76653)),'B':int(os.environ.get('B',68142))}
hdr=pd.read_csv('data/nhanes_1440_PAXPREDM.csv',nrows=0); mc=[c for c in hdr.columns if c.startswith('min_')]
df=pd.read_csv('data/nhanes_1440_PAXPREDM.csv',usecols=['SEQN','PAXDAYM']+mc); df=df[df.SEQN.isin(sel.values())]
def person(sq):
    p=df[df.SEQN==sq].sort_values('PAXDAYM'); M=p[mc].fillna(4).to_numpy(dtype=np.int8); days=list(p.PAXDAYM)
    full=np.full((9,1440),4,dtype=np.int8)
    for dd,row in zip(days,M): full[dd-1]=row
    x=full.reshape(-1); eps=m.episodes_cont(x); b=m.boundary_minute(x)
    daysb=[(b+1440*k,b+1440*(k+1)) for k in range(8)]; info=[]
    for lo,hi in daysb:
        seg=x[lo:hi]; worn=np.isin(seg,[1,2]).sum(); de=[(a,e) for a,e in eps if lo<=(a+e)/2<hi]
        info.append(dict(lo=lo,hi=hi,valid=(worn>=960 and len(de)>0),eps=de))
    start=next(i for i in range(len(info)-6) if all(info[j]['valid'] for j in range(i,i+7)))
    return b,info[start:start+7]
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8.5})
fig,axes=plt.subplots(1,2,figsize=(8.6,4.1),dpi=300)
for ax,(lab,sq) in zip(axes,sel.items()):
    b,W=person(sq); union=np.zeros(1440,bool); tot=0
    for i,w in enumerate(W):
        y=7-i
        for a,e in w['eps']:
            s=max(a,w['lo']); en=min(e,w['hi']); tot+=(e-a)
            ax.add_patch(Rectangle(((s-w['lo'])/60,y-0.3),(en-s)/60,0.6,fc='#1b6f9b',ec='none',zorder=3))
            for t in range(a,e): union[(t-b)%1440]=True
    M=tot/7/60; TSR=union.sum()/60; srt=M/TSR*100
    segs=[]; i=0
    while i<1440:
        if union[i]:
            j=i
            while j<1440 and union[j]: j+=1
            segs.append((i/60,j/60)); i=j
        else: i+=1
    for s,e in segs:
        for xx in (s,e): ax.plot([xx,xx],[-1.8,7.6],ls=(0,(3,2)),lw=0.8,color='#c86f2a',zorder=1)
    yT=-0.5; yM=-1.35
    for s,e in segs: ax.add_patch(Rectangle((s,yT-0.2),e-s,0.4,fc='none',ec='#c86f2a',lw=1.0,zorder=3))
    x0=max(segs,key=lambda p:p[1]-p[0])[0]; ax.add_patch(Rectangle((x0,yM-0.2),M,0.4,fc='#1b6f9b',ec='none',zorder=3))
    ax.text(24.4,yT,f'TSR = {TSR:.1f} h',va='center',fontsize=8); ax.text(24.4,yM,f'M = {M:.1f} h',va='center',fontsize=8)
    bh=b/60; ticks=sorted(((6*k-bh)%24) for k in range(4)); labs=[f'{int(round((bh+t)%24)):02d}:00' for t in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(labs); ax.set_xlim(0,24); ax.set_ylim(-1.9,7.8)
    ax.set_yticks(range(7,0,-1)); ax.set_yticklabels([f'Day {i}' for i in range(1,8)]); ax.tick_params(axis='y',length=0)
    ax.axhline(0.35,color='0.75',lw=0.7)
    for s in ['top','right','left']: ax.spines[s].set_visible(False)
    ax.text(-0.12,1.10,lab,transform=ax.transAxes,fontsize=12,fontweight='bold',va='bottom')
    ax.text(0.0,1.02,f'M = {M:.1f} h,  TSR = {TSR:.1f} h',transform=ax.transAxes,fontsize=8.5,va='bottom')
    ax.text(1.0,1.02,f'SRT = {srt:.0f}',transform=ax.transAxes,fontsize=11,fontweight='bold',va='bottom',ha='right')
    ax.set_xlabel('Clock time')
fig.text(0.5,-0.01,'SRT = M / TSR × 100;  M, mean daily sleep duration (all episodes);  TSR, clock time occupied by sleep on at least one of the 7 days (dashed bounds).',ha='center',fontsize=8)
plt.tight_layout(w_pad=3.5,rect=(0,0,1,0.97)); plt.savefig(''+os.environ.get('OUTP','out/Fig1_concept')+'.png',dpi=300,bbox_inches='tight'); plt.savefig(''+os.environ.get('OUTP','out/Fig1_concept')+'.tiff',dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'}); print('ok')
