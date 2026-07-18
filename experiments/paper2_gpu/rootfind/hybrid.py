"""Validate the production recipe batched_roots (Aberth + eigvals fallback):
fallback rate, selected-power accuracy vs numpy, and timing, on real FTP data;
plus direct Aberth-vs-numpy root agreement on adversarial synthetic cases."""
import os, time
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import rootfinders as rf

HERE = os.path.dirname(os.path.abspath(__file__))
def loadftp(tag):
    d = np.load(os.path.join(HERE, "data", f"{tag}.npz"))
    return d["coefs"], d["YM"], d["MM"], float(d["YY"]), int(d["H"])

print("=== batched_roots (Aberth + eigvals fallback) on real FTP data ===")
print(f"{'case':<14}{'D':>4}{'nf':>7}{'n_fallback':>11}{'pw_relmax':>11}"
      f"{'pw_relmed':>11}{'time(s)':>9}{'speedup':>8}")
for tag in ("ftp_H8_N30","ftp_H8_N60","ftp_H8_N12","ftp_H6_N30","ftp_H4_N30"):
    coefs,YM,MM,YY,H = loadftp(tag); D=coefs.shape[1]-1; nf=len(coefs)
    ref = rf.numpy_roots_loop(coefs)
    refpw,_ = rf.select_power(ref, YM, MM, YY)
    # timing
    rf.batched_roots(coefs)  # warmup
    t0=time.perf_counter(); W,nfb = rf.batched_roots(coefs); dt=time.perf_counter()-t0
    tn0=time.perf_counter(); rf.numpy_roots_loop(coefs); tn=time.perf_counter()-tn0
    pw,_ = rf.select_power(W, YM, MM, YY)
    rel = np.abs(pw-refpw)/np.maximum(np.abs(refpw),1e-300)
    print(f"{tag:<14}{D:>4}{nf:>7}{nfb:>11}{rel.max():>11.2e}"
          f"{np.median(rel):>11.2e}{dt:>9.3f}{tn/dt:>7.1f}x")

print("\n=== adversarial synthetic: does the fallback keep it exact-as-numpy? ===")
print(f"{'case':<16}{'n_fallback/nf':>16}{'aberth-only vs numpy':>22}"
      f"{'hybrid vs numpy':>18}")
for kind in ("clustered","multiple","unit"):
    d = np.load(os.path.join(HERE, "data", f"synth_{kind}_D46.npz"))
    coefs = d["coefs"]; nf=len(coefs)
    npy = rf.numpy_roots_loop(coefs)
    npy_s = np.sort_complex(npy)
    # aberth only
    Wab = rf.batched_aberth(coefs, tol=1e-12, max_iter=60)
    ab_only = np.abs(np.sort_complex(Wab)-npy_s).max()
    # hybrid
    Wh, nfb = rf.batched_roots(coefs)
    hyb = np.abs(np.sort_complex(Wh)-npy_s).max()
    print(f"{kind+' D46':<16}{f'{nfb}/{nf}':>16}{ab_only:>22.2e}{hyb:>18.2e}")
