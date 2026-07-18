"""Throughput benchmark: many sparse griz light curves through the batched
floating_offsets FTP, on CPU (numpy) or GPU (cupy). Reports light curves/sec."""
import sys, time, argparse
import numpy as np
import gpu_ftp_proto as G

def make_batch(xp, B, K, Nper, H, nfreq, seed=0):
    rng = np.random.default_rng(seed)
    n = np.arange(1, H+1); cn=(0.4/n)*np.cos(1.3*n); sn=(0.4/n)*np.sin(1.3*n)
    baseline=1826.0; ft=1/0.55
    bands=['g','r','i','z','y','u','v','x'][:K]
    band_arrays=[]
    # shared freq grid (integer multiples of df so it's a valid RRL grid)
    df=1.0/(baseline*5); i0=int(1.0/df); freqs=(i0+np.arange(nfreq))*df
    Wsum=np.zeros(B)
    # first pass compute per-source global ybar/YY across bands (approx: per-band independent noise)
    allt={}; ally={}; allw={}
    for b,band in enumerate(bands):
        N=Nper
        t=np.sort(rng.uniform(0,baseline,(B,N)),axis=1)
        amp=[1.0,.75,.6,.55,.5,.5,.5,.5][b]; off=18.5+0.1*b
        ph=2*np.pi*ft*t; m=np.zeros((B,N))
        for h in range(1,H+1): m+=amp*(cn[h-1]*np.cos(h*ph)+sn[h-1]*np.sin(h*ph))
        sig=0.12
        y=off+m+rng.normal(0,sig,(B,N)); w=np.full((B,N),1.0/sig**2)
        allt[band]=t; ally[band]=y; allw[band]=w
    # global weight normalization across all bands per source
    totw=sum(allw[b].sum(1) for b in bands)  # (B,)
    ybar_g=sum((allw[b]*ally[b]).sum(1) for b in bands)/totw
    # per-band renormalized weights + W_k + combined YY
    YYc=np.zeros(B)
    for band in bands:
        w=allw[band]; y=ally[band]
        Wk=w.sum(1)/totw                          # (B,)
        wk=w/w.sum(1,keepdims=True)               # renorm within band -> sum 1
        ybar_k=(wk*y).sum(1)
        YYk=(wk*(y-ybar_k[:,None])**2).sum(1)
        YYc+=Wk*YYk
        u=wk*(y-ybar_k[:,None])
        band_arrays.append(dict(t=xp.asarray(allt[band]),w=xp.asarray(wk),u=xp.asarray(u),
                                W=xp.asarray(Wk),ybar_global=xp.asarray(ybar_g),YY=xp.asarray(YYc)))
    # fix YY (needs final YYc) — overwrite
    for bd in band_arrays: bd['YY']=xp.asarray(YYc)
    return band_arrays,(cn,sn),freqs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--backend',choices=['numpy','cupy'],default='numpy')
    ap.add_argument('--B',type=int,default=64); ap.add_argument('--K',type=int,default=4)
    ap.add_argument('--N',type=int,default=15); ap.add_argument('--H',type=int,default=8)
    ap.add_argument('--nfreq',type=int,default=8000); ap.add_argument('--chunk',type=int,default=1024)
    ap.add_argument('--reps',type=int,default=3)
    a=ap.parse_args()
    if a.backend=='cupy':
        import cupy as xp; dev=xp.cuda.Device(); free,tot=dev.mem_info
        print("cupy on",xp.cuda.runtime.getDeviceProperties(0)['name'].decode(),"mem %.0fGB"%(tot/1e9))
    else:
        xp=np; print("numpy CPU")
    band_arrays,tmpl,freqs=make_batch(xp,a.B,a.K,a.N,a.H,a.nfreq)
    freqs_x=xp.asarray(freqs)
    def run(): 
        p=G.floating_offsets_powers(xp,band_arrays,tmpl,freqs_x,chunk=a.chunk,n_newton=8)
        if a.backend=='cupy': xp.cuda.Stream.null.synchronize()
        return p
    run()  # warmup (jit/compile)
    ts=[]
    for _ in range(a.reps):
        t0=time.perf_counter(); p=run(); ts.append(time.perf_counter()-t0)
    tb=min(ts)
    lcs=a.B; lcs_per_s=lcs/tb
    print("B=%d K=%d N=%d H=%d nfreq=%d chunk=%d | best %.3fs | %.1f LC/s | %.1f us/LC/freq | 1e7 LC = %.2f GPU-hr"%(
        a.B,a.K,a.N,a.H,a.nfreq,a.chunk,tb,lcs_per_s,1e6*tb/(a.B*a.nfreq),1e7/lcs_per_s/3600))
if __name__=='__main__': main()
