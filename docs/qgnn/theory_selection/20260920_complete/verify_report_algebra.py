"""Report-only algebra checks: no ICCT/data/model imports, no training."""
import json,math
from pathlib import Path
import numpy as np
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1,-1]).astype(complex)
P={'I':I,'X':X,'Y':Y,'Z':Z}
def op(s):
 a=np.ones((1,1),complex)
 for c in s:a=np.kron(a,P[c])
 return a
def rot(A,t):return np.cos(t/2)*np.eye(A.shape[0])-1j*np.sin(t/2)*A
A=op('ZZI');B=op('IXX');O=op('XII')
y=np.array([1,1j])/np.sqrt(2);p=np.ones(2)/np.sqrt(2);v=np.kron(np.kron(y,y),p)
a,b=.4,.7;ab=rot(B,b)@rot(A,a)@v;ba=rot(A,a)@rot(B,b)@v
f=lambda x:float(np.vdot(x,O@x).real)
fam={k:[] for k in ['Z','X','D1','D2']}
for j in range(3):
 for k in range(3):
  if j==k:continue
  for name,c1,c2 in [('D1','Z','X'),('D2','X','Z')]:
   s=['I']*6;s[2*j]=c1;s[2*k+1]=c2;fam[name].append(op(s))
  if j<k:
   for name in ['X','Z']:
    s=['I']*6;s[2*j]=name;s[2*k]=name;fam[name].append(op(s))
comm={k:max(float(np.max(np.abs(a@b-b@a))) for a in vs for b in vs) for k,vs in fam.items()}
z=np.array([1,1j])/np.sqrt(2);zp=z*np.exp(.37j);u=np.array([1,0],complex);w=np.array([0,1],complex)
resources=[]
for t in [8,12,16]:
 raw=32*t*8*2**12
 resources.append({'targets':t,'candidate_count_is_separate':True,'Q':12,'trajectories_per_batch':32*t,'raw_batch_MiB':raw/2**20,'naive_gate_saved_state_GiB':raw*2496/2**30,'checkpoint16_MiB':raw*16/2**20,'one_bank156_recompute_GiB':raw*156/2**30})
out={'scope':'small algebra and parameter/cost arithmetic only','commutator_error':float(np.max(np.abs(A@B-B@A-2j*op('ZYX')))),'A_then_B_root_X':f(ab),'B_then_A_root_X':f(ba),'analytic_B_then_A':-math.sin(a)*math.sin(b),'family_internal_commutator_errors':comm,'born_global_phase_error':float(abs(np.vdot(z,Z@z)-np.vdot(zp,Z@zp))),'raw_real_amplitude_phase_change':float(np.linalg.norm(z.real-zp.real)),'orthogonal_inputs_inner_product':0.,'after_add_normalize_inner_product':float(abs(np.vdot(u,(w+u)/np.linalg.norm(w+u)))),'quantum_parameters_primary':98,'quantum_parameters_backup':264,'temporal_encoder_parameters':224+6336+64,'TR_TGN_interaction_parameters':23127,'total_trainable_primary':4143895,'total_trainable_backup':4167931,'total_trainable_classical':4166924,'rotation_scheduling_upper_bound':2496,'nontrivial_rotation_slots':2448,'resources':resources,'time_scenarios':[{'assumed_step_seconds':s,'training_minutes':9880*s/60,'total_minutes_assuming_20percent_overhead':9880*s/60*1.2} for s in [.1,.25,.5,1.]]}
assert out['commutator_error']<1e-12 and max(comm.values())<1e-12
assert abs(f(ab))<1e-12 and abs(f(ba)+math.sin(a)*math.sin(b))<1e-12
Path(__file__).with_name('algebra_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
