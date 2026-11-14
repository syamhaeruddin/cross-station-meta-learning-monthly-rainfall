# Leave-group-out transfer with the static descriptors standardized inside each fold.
#
# In meta_gate.build the six static descriptors are standardized once over all fifteen
# stations, so the held-out stations contribute their mean and standard deviation to the
# transfer preprocessing. This script repeats the transfer experiment with the
# standardization refitted on the twelve training stations of each fold only.
#
# Standardization is invariant to an affine change of the inputs, so re-standardizing the
# stored matrix over a subset of stations gives exactly the same result as standardizing
# the raw descriptors over that subset; the held-out stations are then transformed with
# the training-fold mean and standard deviation.
#
#   python src/transfer_outfold.py <fold 0..4>
import os, sys, time
import numpy as np
from common import OUT
from meta_gate import FOLDS, load_ds, station_nnls_prior, fit_predict, nnls_w, prior

f = int(sys.argv[1]); t0 = time.time()
D = load_ds(); LP, W = station_nnls_prior(D)
base_tr = (D["split"] == 0) & D["ok"]; base_va = (D["split"] == 1) & D["ok"]

hold = [list(D["stations"]).index(s) for s in FOLDS[f] if s in D["stations"]]
inh = np.isin(D["st"], hold)
train_st = np.array([i for i in range(len(D["stations"])) if i not in hold])

S = D["static"]
mu, sd = S[train_st].mean(0), S[train_st].std(0)
D["static"] = (S - mu) / (sd + 1e-9)
print(f"fold {f}: standardization refitted on {len(train_st)} training stations; "
      f"held-out {[D['stations'][i] for i in hold]}", flush=True)
print("  held-out descriptors under the training-fold statistics:", flush=True)
for i in hold:
    print("   ", D["stations"][i], np.round(D["static"][i], 3), flush=True)

LPt = np.zeros_like(D["P"])
for j in range(4):
    kt = (~inh) & (D["j"] == j) & (D["split"] < 2) & D["ok"]
    LPt[D["j"] == j] = prior(nnls_w(D["P"][kt], D["y"][kt]))

te = np.where(inh & (D["split"] == 2))[0]
p, w = fit_predict(D, np.where((~inh) & base_tr)[0], np.where((~inh) & base_va)[0],
                   te, True, False, True, LPt, 256, f"transfer out-of-fold {f}")
np.savez(os.path.join(OUT, f"pred_transfer_of_{f}.npz"), idx=te, pred=p, w=w.mean(0))
print(f"fold {f} done", round((time.time() - t0) / 60, 1), "min", flush=True)
