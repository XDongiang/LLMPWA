import numpy as np, tomllib, re, os

root="/home/hyx/LLMPWA/kk_dis"
mass_args=[0,5,11,23,35,47,59]
width_args=[6,12,24,36,48,60]
res_labels={0:'f0_980',5:'f0_1710',6:'f0_1710',11:'f2_1270',12:'f2_1270',23:'f2_1525',24:'f2_1525',35:'f2_2150',36:'f2_2150',47:'f2_2340',48:'f2_2340',59:'f0_2470',60:'f0_2470'}

summ=open(f"{root}/output/multistart/free_scan_d3_summary.toml").read()
nll={}
for m in re.finditer(r'start = (\d+)\n.*?nll = ([-\d.eE]+)', summ, re.S):
    nll[int(m.group(1))]=float(m.group(2))

physical=[]
nonphys=[]
for start in range(35):
    p=f"{root}/work/free_scan_d3_start_{start}/output/fit/fit_result_values.npy"
    if not os.path.exists(p):
        continue
    a=np.load(p)
    bad=False
    details=[]
    for wi in width_args:
        if a[wi]<=0:
            bad=True
            details.append(f"W{wi}={a[wi]:.3f}({res_labels[wi]})")
    for mi in mass_args:
        if not (0.9<=a[mi]<=3.0):
            bad=True
            details.append(f"M{mi}={a[mi]:.3f}({res_labels[mi]})")
    n=nll.get(start,float('nan'))
    rec=(start, n, a, bad, "; ".join(details))
    (nonphys if bad else physical).append(rec)

print(f"=== FREE d3 disturb=3, 35 starts ===")
print(f"physical={len(physical)}  nonphysical={len(nonphys)}")
print("\n--- NON-PHYSICAL starts ---")
for s,n,a,bad,d in sorted(nonphys,key=lambda x:x[0]):
    print(f"start{s:2d} nll={n:9.3f}  {d}")
print("\n--- PHYSICAL starts (sorted by NLL) ---")
for s,n,a,bad,d in sorted(physical,key=lambda x:-x[1]):
    print(f"start{s:2d} nll={n:9.3f}")
