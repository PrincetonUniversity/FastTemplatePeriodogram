import sys; sys.path.insert(0,'experiments/paper2_gpu')
import numpy as np
from ftperiodogram.template import Template
from ftperiodogram.multiband import FastMultibandTemplatePeriodogram, _prepare_bands
import gpu_ftp_proto as G

rng=np.random.default_rng(0); H=8
n=np.arange(1,H+1); cn=(0.4/n)*np.cos(1.3*n); sn=(0.4/n)*np.sin(1.3*n); tmpl=Template(cn,sn)
bands=['g','r','i','z']; nper={'g':12,'r':20,'i':18,'z':4}; baseline=1826.0; ft=1/0.55
amps={'g':1.,'r':.75,'i':.6,'z':.55}; offs={'g':18.9,'r':18.5,'i':18.4,'z':18.6}; sig={'g':.13,'r':.12,'i':.11,'z':.15}
ta=[];ya=[];bb=[];da=[]
for band in bands:
    N=nper[band]; tk=np.sort(rng.uniform(0,baseline,N)); ph=2*np.pi*ft*tk; m=np.zeros(N)
    for h in range(1,H+1): m+=amps[band]*(cn[h-1]*np.cos(h*ph)+sn[h-1]*np.sin(h*ph))
    yk=offs[band]+m+rng.normal(0,sig[band],N); ta.append(tk);ya.append(yk);bb.append([band]*N);da.append(np.full(N,sig[band]))
t=np.concatenate(ta);y=np.concatenate(ya);ba=np.concatenate(bb);dy=np.concatenate(da)
df=1.0/(baseline*5); freqs=np.arange(max(1,int(1.0/df)),int(2.0/df)+1)*df

# package reference
mb=FastMultibandTemplatePeriodogram(templates=tmpl,mode='floating_offsets').fit(t,y,ba,dy)
p_ref=mb.power(freqs,fast=False,save_best_model=False,method='scan')

# build padded band_arrays for the prototype (B=1)
band_data,stats=_prepare_bands(t,y,ba,dy,'floating_offsets',None,H)
Nmax=max(len(v[0]) for v in band_data.values())
band_arrays=[]
for band in stats.bands:
    t_k,y_k,w_k=band_data[band]; N=len(t_k)
    tp=np.zeros((1,Nmax)); wp=np.zeros((1,Nmax)); up=np.zeros((1,Nmax))
    ybar_k=np.dot(w_k,y_k)
    tp[0,:N]=t_k; wp[0,:N]=w_k; up[0,:N]=w_k*(y_k-ybar_k)
    band_arrays.append(dict(t=tp,w=wp,u=up,W=np.array([stats.W[band]]),
                            ybar_global=np.array([stats.ybar_global]),YY=np.array([stats.YY_combined])))
p_gpu=G.floating_offsets_powers(np, band_arrays,(cn,sn),freqs,chunk=4096)[0]

d=np.abs(p_ref-p_gpu)
print("prototype(xp=numpy) vs package floating_offsets powers:")
print("  max|dP|=%.3e  median=%.3e"%(d.max(),np.median(d)))
print("  argmax same: %s (ref f=%.5f, proto f=%.5f)"%(p_ref.argmax()==p_gpu.argmax(),freqs[p_ref.argmax()],freqs[p_gpu.argmax()]))
print("  peak power ref=%.6f proto=%.6f"%(p_ref.max(),p_gpu.max()))
