import numpy as np, re

root="/home/hyx/LLMPWA/kk_dis"
mi=[0,5,11,23,35,47,59]; wi=[6,12,24,36,48,60]
names=['f0_980','f0_1710','f2_1270','f2_1525','f2_2150','f2_2340','f0_2470']

def parse(tag):
    t=open(f"{root}/output/multistart/{tag}_summary.toml").read()
    nll={}
    for b in re.split(r'\[\[results\]\]', t):
        m=re.search(r'start = (\d+)', b); n=re.search(r'nll = ([-\d.eE]+)', b)
        if m and n: nll[int(m.group(1))]=float(n.group(1))
    return nll

def classify(tag):
    nll=parse(tag)
    phys=[]; negw=[]; massbad=[]
    for s,n in nll.items():
        a=np.load(f"{root}/work/{tag}_start_{s}/output/fit/fit_result_values.npy")
        nw=[names[i] for i in [1,2,3,4,5,6] if a[wi[i-1]]<=0]
        mb=[(names[i],a[mi[i]]) for i in range(7) if not(0.9<=a[mi[i]]<=3.0)]
        if nw: negw.append((s,n,nw,mb))
        elif mb: massbad.append((s,n,mb))
        else: phys.append((s,n))
    return phys,negw,massbad

for tag in ['free_scan_d3','range_on_scan_d3']:
    phys,negw,massbad=classify(tag)
    total=len(phys)+len(negw)+len(massbad)
    print(f"\n===== {tag} =====")
    print(f"success={total}  physical={len(phys)}  mass-out-of-band={len(massbad)}  negative-width={len(negw)}")
    if phys:
        bp=min(phys,key=lambda x:x[1]); print(f"  BEST physical: start{bp[0]} nll={bp[1]:.3f}")
    if negw:
        bn=min(negw,key=lambda x:x[1])
        print(f"  BEST negative-width: start{bn[0]} nll={bn[1]:.3f} widths={bn[2]}")
    for s,n,nw,mb in sorted(negw,key=lambda x:x[1]):
        print(f"    negwidth start{s:2d} nll={n:9.3f} {nw}")
