import time, random
import numpy as np
from sponge_ref import *
from render import frame_look, slice_basis, zoom_in
rng=np.random.default_rng(5)
c=np.array([.5,.5,.5,.5]); B=slice_basis([1,2,3,4]); eye=c+B@np.array([1.5,1.1,-1.9]); F3=frame_look(eye,c+B@np.array([.11,.07,0]),[0,1,0],B)
pos,_=zoom_in(eye,F3,4,2,60)
pix=2*np.tan(np.radians(25))/1080
t0=time.perf_counter(); de=normalize(pos,4,2); tb=level_table(pos,4,2,far=1000.0,fine_min=0.3*de*pix); t1=time.perf_counter()
rows=table_rows(tb)
print("E=%d levels=%d  normalize+table %.2f ms"%(pos.E,len(tb.w),(t1-t0)*1e3))
# scalar equals vector
q=rng.normal(0,1,(2000,4))*(10.0**rng.uniform(-3,2.5,2000))[:,None]
dv=de_table(q,tb,dtype=np.float64)
t0=time.perf_counter(); ds=np.array([de_scalar(list(r),rows) for r in q]); t1=time.perf_counter()
print("scalar vs vector max abs diff %.1e ; scalar DE %.1f us/point"%(np.abs(dv-ds).max(),(t1-t0)/2000*1e6))
# rays
ts=[]; t0=time.perf_counter()
for i in range(200):
    d=rng.normal(0,1,3); d/=np.linalg.norm(d); D=F3@d
    ts.append(free_distance(rows,list(D),pix,1000.0)[0])
t1=time.perf_counter()
print("free_distance: %.2f ms per ray (mean over 200 random in-slice rays), distances %.2f..%.1f"%((t1-t0)/200*1e3,min(ts),max(ts)))
# drunk fly: random moves limited to half the free distance never end in solid
random.seed(3); p=pos.copy(); Emin=Emax=p.E; t0=time.perf_counter(); n_moves=1500
for i in range(n_moves):
    de=normalize(p,4,2); assert de>0
    tb=level_table(p,4,2,far=1000.0,fine_min=0.3*de*pix); rows=table_rows(tb)
    u=rng.normal(0,1,4); u/=np.linalg.norm(u)               # any 4D direction, including out of the slice
    fd,_=free_distance(rows,list(u),pix,1000.0)
    step=min(random.choice((0.3,1.0,3.0,30.0))*de, 0.5*fd)
    p.move(u*step); Emin=min(Emin,p.E); Emax=max(Emax,p.E)
print("drunk fly: %d moves, never in solid, E ranged %d..%d, %.2f ms per move (normalize+table+ray)"%(n_moves,Emin,Emax,(time.perf_counter()-t0)/n_moves*1e3))
