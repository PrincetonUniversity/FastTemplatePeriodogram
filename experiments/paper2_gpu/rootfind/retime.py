"""Re-time Aberth (inflate=1.2) + D-scaling + cross-case robustness."""
import os, time
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import rootfinders as rf

HERE = os.path.dirname(os.path.abspath(__file__))
def load(tag):
    d = np.load(os.path.join(HERE, "data", f"{tag}.npz"))
    return d["coefs"], d["YM"], d["MM"], float(d["YY"]), int(d["H"])
def chunked(fn, co, ch=4096, **k):
    if len(co)<=ch: return fn(co, **k)
    return np.concatenate([fn(co[i:i+ch],**k) for i in range(0,len(co),ch)],0)
def chunked_stats(fn, co, ch=4096, **k):
    Ws,its=[],[]
    for i in range(0,len(co),ch):
        W,it=fn(co[i:i+ch],return_stats=True,**k); Ws.append(W); its.append(it)
    return np.concatenate(Ws,0), np.concatenate(its,0)
def T(f, r=3):
    f()  # warmup
    return min(_t(f) for _ in range(r))
def _t(f):
    t=time.perf_counter(); f(); return time.perf_counter()-t

print("=== D-scaling: numpy loop vs Aberth vs batched-eigvals, nf=20000 ===")
print(f"{'D':>4}{'numpy(s)':>10}{'aberth(s)':>11}{'speedup':>9}"
      f"{'eigvals(s)':>12}{'speedup':>9}{'ab_med_it':>10}")
for tag in ("ftp_H4_N30","ftp_H6_N30","ftp_H8_N30"):
    coefs,YM,MM,YY,H = load(tag); D=coefs.shape[1]-1
    tn = T(lambda: rf.numpy_roots_loop(coefs))
    ta = T(lambda: chunked(rf.batched_aberth, coefs, max_iter=60))
    te = T(lambda: chunked(rf.batched_companion_eigvals, coefs))
    _,its = chunked_stats(rf.batched_aberth, coefs, max_iter=60)
    print(f"{D:>4}{tn:>10.3f}{ta:>11.3f}{tn/ta:>8.1f}x{te:>12.3f}{tn/te:>8.1f}x"
          f"{int(np.median(its)):>10}")

print("\n=== nf-scaling at D=46 (H=8) ===")
coefs,YM,MM,YY,H = load("ftp_H8_N30")
print(f"{'nf':>7}{'numpy(s)':>10}{'aberth(s)':>11}{'speedup':>9}")
for nf in (1000,5000,20000):
    c=coefs[:nf]
    tn=T(lambda: rf.numpy_roots_loop(c))
    ta=T(lambda: chunked(rf.batched_aberth, c, max_iter=60))
    print(f"{nf:>7}{tn:>10.3f}{ta:>11.3f}{tn/ta:>8.1f}x")

print("\n=== robustness: Aberth accuracy + iters across all FTP cases ===")
print(f"{'case':<14}{'D':>4}{'nonconv':>8}{'med_it':>7}{'max_it':>7}"
      f"{'pw_relmax':>11}{'pw_relp99':>11}")
for tag in ("ftp_H8_N30","ftp_H8_N60","ftp_H8_N12","ftp_H6_N30",
            "ftp_H4_N30","ftp_H4_N60","ftp_H6_N60"):
    coefs,YM,MM,YY,H = load(tag); D=coefs.shape[1]-1
    ref = rf.numpy_roots_loop(coefs)
    refpw,_ = rf.select_power(ref, YM, MM, YY)
    W,its = chunked_stats(rf.batched_aberth, coefs, max_iter=60)
    conv = its<60
    pw,_ = rf.select_power(W, YM, MM, YY)
    rel = np.abs(pw-refpw)/np.maximum(np.abs(refpw),1e-300)
    print(f"{tag:<14}{D:>4}{int((~conv).sum()):>8}{int(np.median(its[conv])):>7}"
          f"{int(its[conv].max()):>7}{rel.max():>11.2e}{np.percentile(rel,99):>11.2e}")
