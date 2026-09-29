import numpy as np, re, os

root="/home/hyx/LLMPWA/kk_dis"
mi=[0,5,11,23,35,47,59]; wi=[6,12,24,36,48,60]
labels={0:'f0_980',5:'f0_1710',11:'f2_1270',23:'f2_1525',35:'f2_2150',47:'f2_2340',59:'f0_2470'}

def classify(tag):
    summ=open(f"{root}/output/multistart/{tag}_summary.toml").read()
    nll={}
    for m in re.finditer(r'start = (\d+)\n.*?nll = ([-\d.eE]+)', summ, re.S):
        nll[int(m.group(1))]=float(m.group(2))
    starts=set(int(m.group(1)) for m in re.finditer(r'start = (\d+)', summ))
    negw=[]; massbad=[]; phys=[]; total=0
    for s in sorted(starts):
        p=f"{root}/work/{tag}_start_{s}/output/fit/fit_result_values.npy"
        if not os.path.exists(p): continue
        a=np.load(p); total+=1
        n=nll.get(s,float('nan'))
        nw=[l for i,l in zip(wi,['f0_1710','f2_1270','f2_1525','f2_2150','f2_2340','f0_2470']) if a[i]<=0]
        mb=[(l,a[i]) for i,l in zip(mi,['f0_980','f0_1710','f2_1270','f2_1525','f2_2150','f2_2340','f0_2470']) if not(0.9<=a[i]<=3.0)]
        if nw: negw.append((s,n,nw))
        elif mb: massbad.append((s,n,mb))
        else: phys.append((s,n))
    print(f"\n===== {tag} ===== total_success={total}")
    print(f"  physical: {len(phys)}   mass-out-of-band but no-neg-width: {len(massbad)}   negative-width: {len(negw)}")
    print(f"  -> non-physical total = {len(negw)+len(massbad)} ({100*(len(negw)+len(massbad))/total:.0f}%)")
    if negw:
        best=max(negw,key=lambda x:x[1])
        print(f"  best(non-phys,neg-width) NLL={best[1]:.3f} start{best[0]} widths={best[2]}")
    if phys:
        bp=max(phys,key=lambda x:x[1])
        print(f"  best(physical) NLL={bp[1]:.3f} start{bp[0]}")
    return dict(phys=len(phys),negw=len(negw),massbad=len(massbad),total=total,
                bestphys=max(phys,key=lambda x:x[1]) if phys else None,
                bestneg=max(negw,key=lambda x:x[1]) if negw else None)

f=classify('free_scan_d3')
r=classify('range_on_scan_d3')
print("\n=== COMPARISON ===")
print(f"FREE      non-phys {f['negw']+f['massbad']}/{f['total']} ({100*(f['negw']+f['massbad'])/f['total']:.0f}%)  neg-width={f['negw']}  best-phys={f['bestphys'][1]:.2f}  best-nonphys={f['bestneg'][1]:.2f}")
print(f"RANGE-ON  non-phys {r['negw']+r['massbad']}/{r['total']} ({100*(r['negw']+r['massbad'])/r['total']:.0f}%)  neg-width={r['negw']}  best-phys={r['bestphys'][1]:.2f}  best-nonphys={r['bestneg'][1]}")
