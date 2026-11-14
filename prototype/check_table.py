import random, time
from fractions import Fraction as F
import numpy as np
from sponge_ref import *
rng = np.random.default_rng(2); random.seed(2)

def rand_pos(n, E, mode):
    O=[]; 
    for i in range(n):
        if mode=="random": digs=[random.randrange(3) for _ in range(E)]
        elif mode=="ones": digs=[1]*E
        elif mode=="cantor": digs=[random.choice((0,2)) for _ in range(E)]
        elif mode=="wall":   digs=[1]+[0]*(E-1) if i<2 else [random.randrange(3) for _ in range(E)]
        v=0
        for d in digs: v=3*v+d
        O.append(v)
    f=rng.uniform(0,1,n)
    if mode=="wall": f[:2]=rng.uniform(1e-9,1e-3,2)
    return Pos(O,f,E)

def exact_local(pos, q, n, m, L):
    X=pos.exact()
    s=F(3)**pos.E
    out=[]
    for row in q:
        x=[X[i]+F(float(row[i]))/s for i in range(n)]
        out.append(float(de_exact(x,m,L)*s))
    return np.array(out)

worst=0; compared={}; skipped=[]
t0=time.time()
for n,m in [(2,2),(3,2),(4,2),(4,3),(3,1),(4,4)]:
    for E in (0,1,5,20,60,200):
        for mode in ("random","ones","cantor","wall"):
            if E==0 and mode!="random": continue
            pos=rand_pos(n,E,mode)
            K=8
            tb=level_table(pos,n,m,max_level=E+K)       # no pruning, K fine levels
            if tb.dropped: skipped.append((n,m,E,mode)); continue     # more than 64 rows: only a camera inside holes of 30+ levels at once
            compared[E]=compared.get(E,0)+1
            N=40
            scale=10.0**rng.uniform(-3,min(4,0.45*E+0.01) if E>0 else -0.5,N)
            q=np.zeros((N,4)); q[:,:n]=rng.normal(0,1,(N,n))*scale[:,None]
            q=q.astype(np.float32)
            d32=de_table(q,tb)
            dex=np.clip(exact_local(pos,q,n,m,E+K),-BIG,BIG)      # the table clamps distances at 1e30 local units
            # stay inside the cube for the exact comparison (box outside term needs sqrt)
            X=pos.exact(); s=F(3)**E
            ins=np.array([all(0<X[i]+F(float(r[i]))/s<1 for i in range(n)) for r in q])
            err=np.abs(d32-dex)[ins]/(np.linalg.norm(q,axis=1)[ins]+np.abs(dex[ins])+1e-30)
            if ins.any(): worst=max(worst,err.max())
            assert (err<2e-5).all(), (n,m,E,mode,err.max())
            # sign agreement away from the surface
            far_from_surface=np.abs(dex[ins])>1e-4*(np.linalg.norm(q,axis=1)[ins])
            assert (np.sign(d32[ins])==np.sign(dex[ins]))[far_from_surface].all()
print("T3 float32 table vs exact rationals: worst relative error %.2e  (%.1fs); cases compared per depth %s; skipped (over 64 rows) %s"%(worst,time.time()-t0,compared,skipped))

# T4 rebase invariance + position arithmetic
def norm_pos(n,m,E,mode):
    while True:
        pos=rand_pos(n,E,mode)
        try:
            normalize(pos,n,m); return pos
        except ValueError: pass
for n in (2,3,4):
    for E in (3,30,150):
        pos=rand_pos(n,E,"random"); X0=pos.exact()
        p2=pos.copy(); p2.rescale(+2); assert max(abs(float((a-b)*F(3)**E)) for a,b in zip(p2.exact(),X0))<1e-15
        p3=pos.copy(); p3.rescale(-1)
        assert max(abs(float((a-b)*F(3)**E)) for a,b in zip(p3.exact(),X0))<1e-15
        mv=rng.normal(0,5,n); p4=pos.copy(); p4.move(mv); p4.move(-mv)
        assert max(abs(float((a-b)*F(3)**E)) for a,b in zip(p4.exact(),X0))<1e-13
        pos=norm_pos(n,2,E,"random"); p2=pos.copy(); p2.rescale(+2)
        tb=level_table(pos,n,2,far=1e3,fine_min=1e-3); tb2=level_table(p2,n,2,far=9e3,fine_min=9e-3)
        assert list(tb.level)==list(tb2.level)
        q=np.zeros((300,4),dtype=np.float32); q[:,:n]=rng.normal(0,1,(300,n))*(10.0**rng.uniform(-2,2,300))[:,None]
        d1=de_table(q,tb); d2=de_table(q*np.float32(9),tb2)/9
        assert np.allclose(d1,d2,rtol=2e-4,atol=2e-6*np.linalg.norm(q,axis=1).max()), np.abs(d1-d2).max()
print("T4 rescale/move exact, rebase-invariant DE ok")

# T5 pruning validity: far-limited table equals the full table for points within far
cnt=[]
for n,m in [(3,2),(4,2),(4,3)]:
    for E in (10,40,62,300):
        for mode in ("random","cantor","ones","wall"):
            pos=norm_pos(n,m,E,mode)
            far=1000.0
            full=level_table(pos,n,m,fine_min=1e-3)
            pr=level_table(pos,n,m,far=far,fine_min=1e-3)
            cnt.append((n,E,mode,pos.E,len(full.w)+full.dropped,len(pr.w)+pr.dropped))
            if full.dropped: continue
            q=np.zeros((500,4),dtype=np.float32); v=rng.normal(0,1,(500,n)); v/=np.linalg.norm(v,axis=1)[:,None]
            q[:,:n]=v*rng.uniform(0,far,(500,1))
            a=de_table(q,full); b=de_table(q,pr)
            pos_=a>0
            assert np.array_equal(a[pos_],b[pos_]) and ((a>0)==(b>0)).all(), (n,m,E,mode)
print("T5 pruning ok; (n, E asked, mode, E after normalize, levels without far limit, with far=1000):")
for c in cnt: print("   ",c)
