"""Independent exact product-input Pauli moments, T04; no state-vector gates.

Hadamards on V turn the commuting ZX graph into a diagonal ZZ graph.
Only flipped spins are enumerated; all other spins are summed analytically.
This formula is valid for the FIRST graph layer only.
"""
import itertools
import torch


def dense_circuit(encoding_angles, graph_angles, mixing):
    """Small N<=3 independent spectral-exponential reference for gate order.

    Pure vector only; dense matrices are operators, never density matrices.
    """
    n=encoding_angles.shape[-2]
    if n>3:raise ValueError('Dense reference is restricted to <=3 vehicles')
    device=encoding_angles.device
    matrices={
        'I':torch.eye(2,device=device,dtype=torch.complex128),
        'X':torch.tensor([[0,1],[1,0]],device=device,dtype=torch.complex128),
        'Y':torch.tensor([[0,-1j],[1j,0]],device=device,dtype=torch.complex128),
        'Z':torch.tensor([[1,0],[0,-1]],device=device,dtype=torch.complex128)}
    def operator(ops):
        out=torch.ones(1,1,device=device,dtype=torch.complex128)
        for k in range(2*n):out=torch.kron(out,matrices[ops.get(k,'I')])
        return out
    def rotation(ops,angle):
        eig,vec=torch.linalg.eigh(operator(ops))
        return (vec*torch.exp(-.5j*angle*eig))@vec.conj().T
    state=torch.zeros(4**n,device=device,dtype=torch.complex128);state[0]=1
    for l in range(len(encoding_angles)):
        for i in range(n):
            for reg in range(2):
                for offset,p in ((0,'Y'),(1,'Z')):
                    state=rotation({2*i+reg:p},encoding_angles[l,i,2*reg+offset])@state
        for j in range(n):
            for i in range(n):
                if j!=i:state=rotation({2*j:'Z',2*i+1:'X'},graph_angles[l,j,i])@state
        ap,bp,av,bv,eta=mixing[l]
        for i in range(n):
            for ops,angle in (({2*i:'X',2*i+1:'X'},eta),({2*i:'Y'},bp),({2*i:'Z'},ap),({2*i+1:'Y'},bv),({2*i+1:'Z'},av)):
                state=rotation(ops,angle)@state
    return state


def product_amplitudes(angles):
    """angles[N,4] = thetaP,phiP,thetaV,phiV (actual centered angles)."""
    theta, phi = angles.reshape(-1, 2).unbind(-1)
    a = torch.stack((torch.cos(theta/2)*torch.exp(-.5j*phi),
                     torch.sin(theta/2)*torch.exp(.5j*phi)), -1)
    h = torch.tensor([[1, 1], [1, -1]], device=a.device, dtype=a.dtype)/2**.5
    return torch.stack([v if k % 2 == 0 else h@v for k, v in enumerate(a)])


def exact_moment(angles, graph_angles, operators):
    """Single graph, source,target g; operators={qubit:'X'/'Y'/'Z'}."""
    a = product_amplitudes(angles)
    n = angles.shape[0]
    ops, sign = {}, 1
    for k, op in operators.items():
        if k % 2:
            sign *= -1 if op == 'Y' else 1
            op = {'X':'Z', 'Y':'Y', 'Z':'X'}[op]
        ops[k] = op
    # Assemble the undirected ZZ coefficients without detaching gradients.
    coupling = torch.stack([torch.stack([
        graph_angles[u//2, v//2] if u % 2 == 0 and v % 2 else
        graph_angles[v//2, u//2] if u % 2 and v % 2 == 0 else
        graph_angles.sum()*0 for v in range(2*n)]) for u in range(2*n)])
    flipped = [k for k, op in ops.items() if op in ('X', 'Y')]
    fixed = [k for k in range(2*n) if k not in flipped]
    result = a.sum()*0
    for bits in itertools.product((0, 1), repeat=len(flipped)):
        term = a.sum()*0+1
        for k, bit in zip(flipped, bits):
            z = 1-2*bit
            term = term*a[k, 1-bit].conj()*a[k, bit]
            if ops[k] == 'Y':
                term = term*(1j*z)
        for k in fixed:
            field = sum((coupling[f,k]*(1-2*b) for f,b in zip(flipped,bits)),
                        graph_angles.sum()*0)
            u = (a[k,0].abs()**2-a[k,1].abs()**2)
            factor = (u*torch.cos(field)-1j*torch.sin(field) if ops.get(k) == 'Z'
                      else torch.cos(field)-1j*u*torch.sin(field))
            term = term*factor
        result = result+term
    return sign*result.real


def first_layer_moments(angles, graph_angles):
    """Return singles[N,6], raw/connected[N,N,4], local[N,2]."""
    n = angles.shape[0]
    singles = torch.stack([torch.stack([exact_moment(angles, graph_angles, {2*i+q:o})
                                       for q in range(2) for o in 'XYZ']) for i in range(n)])
    pairs = [('Z','X'),('Z','Y'),('Z','Z'),('X','Z')]
    raw = torch.stack([torch.stack([torch.stack([
        exact_moment(angles, graph_angles, {2*j:p,2*i+1:v}) for p,v in pairs
    ]) for i in range(n)]) for j in range(n)])
    connected = raw-torch.stack([torch.stack([torch.stack([
        singles[j,'XYZ'.index(p)]*singles[i,3+'XYZ'.index(v)] for p,v in pairs
    ]) for i in range(n)]) for j in range(n)])
    local = torch.stack([torch.stack([
        exact_moment(angles, graph_angles, {2*i:p,2*i+1:p})-
        singles[i,'XYZ'.index(p)]*singles[i,3+'XYZ'.index(p)] for p in 'XY'
    ]) for i in range(n)])
    return dict(singles=singles,raw=raw,connected=connected,local=local)
