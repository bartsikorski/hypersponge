import itertools, random, time
from fractions import Fraction as F
import numpy as np
from sponge_ref import *
rng = np.random.default_rng(1); random.seed(1)

# T1 counts
for n in (2,3,4):
    for m in range(1,n+1):
        c = sum(1 for d in itertools.product(range(3), repeat=n) if sum(x==1 for x in d) < m)
        assert c == kept_cells(n,m), (n,m,c)
print("T1 counts ok", [(n,m,kept_cells(n,m)) for n in (2,3,4) for m in range(1,n+1)])

# T2 sign + lower bound + Lipschitz versus brute force at level L
def cells(n,m,L):
    cs=[np.zeros(n)]; size=1.0
    for _ in range(L):
        size/=3
        new=[]
        for c in cs:
            for d in itertools.product(range(3), repeat=n):
                if sum(x==1 for x in d) < m: new.append(c+np.array(d)*size)
        cs=new
    return np.array(cs), size
def true_dist(p, cs, size):
    # distance from points p (N,n) to union of boxes [c, c+size]
    out=np.empty(len(p))
    for i in range(0,len(p),256):
        pp=p[i:i+256,None,:]
        dd=np.maximum(np.maximum(cs[None]-pp, pp-(cs[None]+size)),0)
        out[i:i+256]=np.sqrt((dd**2).sum(-1)).min(1)
    return out
for (n,m,L) in [(2,2,3),(2,1,3),(3,2,2),(3,2,3),(3,3,2),(3,1,2),(4,2,2),(4,3,2),(4,4,2),(4,1,2)]:
    cs,size=cells(n,m,L)
    N=3000
    p=rng.uniform(-0.3,1.3,(N,n))
    td=true_dist(p,cs,size)
    pos=Pos.from_floats([0.0]*n,0)            # camera at the origin corner, E=0
    tb=level_table(pos,n,m,max_level=L)
    q=np.zeros((N,4)); q[:,:n]=p
    d=de_table(q,tb,dtype=np.float64)
    inside = td==0
    assert np.all(d[inside] <= 1e-12), "sign in"
    assert np.all(d[~inside] > -1e-12), "sign out"
    assert np.all(d[~inside] <= td[~inside]+1e-9), ("lower bound", (d-td)[~inside].max())
    ratio = d[~inside]/td[~inside]
    # Lipschitz
    p2 = p + rng.normal(0,0.01,(N,n)); q2=np.zeros((N,4)); q2[:,:n]=p2
    d2=de_table(q2,tb,dtype=np.float64)
    lip = np.abs(d2-d)/np.linalg.norm(p2-p,axis=1)
    out = ~inside & (true_dist(p2,cs,size)>0)
    print(f"T2 n={n} m={m} L={L}: cells={len(cs)} lower-bound ok, DE/true median={np.median(ratio):.2f} min={ratio.min():.3f}, Lipschitz max(outside)={lip[out].max():.3f}")
