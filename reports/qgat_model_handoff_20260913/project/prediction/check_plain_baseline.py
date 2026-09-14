"""Current GNN graph-core acceptance against independently loaded source layers."""
import json
from pathlib import Path
import torch
from .classical import GNNGraph, PLAIN_SOURCE

ROOT = Path(__file__).resolve().parents[1]


def run(device='cuda:1'):
    torch.manual_seed(20260912)
    checks = {}
    def record(name, passed, **evidence):
        checks[name] = dict(passed=bool(passed), **evidence)
        print(json.dumps({name: checks[name]}), flush=True)
        assert passed, name
    namespace = {'__name__': __name__}
    exec(compile(PLAIN_SOURCE.read_text(), str(PLAIN_SOURCE), 'exec'), namespace)
    original = torch.nn.ModuleList(namespace['DenseEdgeGraphAttention'](128,4,.1) for _ in range(2)).double().to(device).eval()
    graph = GNNGraph().double().to(device).eval()
    with torch.no_grad():
        for layer in original:
            layer.message_gate.fill_(.37)
            layer.feed_forward_gate.fill_(-.23)
    graph.graph_layers.load_state_dict(original.state_dict())
    x = (torch.randn(2,3,4,4,device=device,dtype=torch.float64)*3).requires_grad_()
    z = torch.randn_like(x, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    zr = z.detach().clone().requires_grad_()
    mask = torch.ones(2,3,4,device=device,dtype=torch.bool)
    mask[0,1,2] = False
    detected = mask.clone(); detected[1,1,1]=False
    a = graph(x,z,mask,detected)
    clean = torch.where(mask[...,None],y,0).reshape(-1,4,4)
    node_input = torch.cat((torch.where(mask[...,None],zr,0),detected[...,None].double(),mask[...,None].double()),-1)
    h = (graph.input(node_input)*mask[...,None]).reshape(-1,4,128)
    edge,distance = namespace['build_edge_features'](clean[...,:2],clean[...,2:])
    flat_mask=mask.reshape(-1,4)
    adjacency=flat_mask[:,:,None]&flat_mask[:,None,:]&(distance<=45)
    for layer in original:
        h,_=layer(h,edge,adjacency)
        h=h*flat_mask[...,None]
    ref=h.reshape_as(a)
    w=torch.randn_like(a)
    ga=torch.autograd.grad((a*w).sum(),(x,z,*graph.input.parameters(),*graph.graph_layers.parameters()))
    gr=torch.autograd.grad((ref*w).sum(),(y,zr,*graph.input.parameters(),*original.parameters()))
    errors=[(u-v).abs().max().item() for u,v in zip(ga,gr)]
    output_error=(a-ref).abs().max().item()
    record('graph_core_source_output_and_gradients',output_error<1e-10 and max(errors)<1e-8,
           output_max_abs=output_error,physical_gradient_max_abs=errors[0],standardized_gradient_max_abs=errors[1],
           all_parameter_gradient_max_abs=max(errors[2:]),parameter_tensors=len(errors)-2,
           nonzero_gates_exercised=True,graph_layers=2,graph_heads=4,hidden_dim=128,edge_dim=7)
    # Independent explicit SI edge/radius oracle, including closing and receding pairs.
    p=torch.tensor([[[0.,0.],[45.,0.],[45.01,0.]]],device=device,dtype=torch.float64)
    v=torch.tensor([[[2.,0.],[0.,0.],[4.,0.]]],device=device,dtype=torch.float64)
    edge,dist=namespace['build_edge_features'](p,v)
    expected=torch.tensor([1.5,0.,-2/15,0.,1.,2/15,1.],device=device,dtype=torch.float64)
    # TTC=22.5 seconds clips to 20 seconds for the 45 m / 2 m/s pair.
    edge_error=(edge[0,0,1]-expected).abs().max().item()
    captured={}
    def capture(module,args):
        captured['adjacency']=args[2].detach().clone()
        captured['edge']=args[1].detach().clone()
    hook=graph.graph_layers[0].register_forward_pre_hook(capture)
    fixture=torch.cat((p,v),-1)
    fm=torch.ones(1,3,device=device,dtype=torch.bool)
    graph(fixture,fixture/20,fm,fm)
    hook.remove()
    adj=captured['adjacency'][0]
    record('seven_SI_edges_radius_and_self_loops',edge_error<1e-12 and adj[0,1] and not adj[0,2]
           and adj.diagonal().all() and captured['edge'].shape[-1]==7,
           directed_edge_max_abs=edge_error,within_45m=True,outside_45m_excluded=True,self_loops=True)
    x=x.detach().requires_grad_(); z=z.detach().requires_grad_()
    out=graph(x,z,mask,detected)
    grad=torch.autograd.grad(out.square().sum(),(x,z))
    perm=torch.tensor([2,0,3,1],device=device)
    xp=x.detach()[:,:,perm].requires_grad_(); zp=z.detach()[:,:,perm].requires_grad_()
    reordered=graph(xp,zp,mask[:,:,perm],detected[:,:,perm])
    pg=torch.autograd.grad(reordered.square().sum(),(xp,zp))
    pe=(reordered-out.detach()[:,:,perm]).abs().max().item()
    ge=max((u-v[:,:,perm]).abs().max().item() for u,v in zip(pg,grad))
    badx=torch.where(mask[...,None],x.detach(),float('nan'))
    badz=torch.where(mask[...,None],z.detach(),float('nan'))
    bad=graph(badx,badz,mask,detected)
    padding=torch.full_like(x,float('nan'))
    pm=torch.cat((mask,torch.zeros_like(mask)),2)
    pd=torch.cat((detected,torch.ones_like(detected)),2)
    padded=graph(torch.cat((x.detach(),padding),2),torch.cat((z.detach(),padding),2),pm,pd)
    paderr=(padded[:,:,:4]-out.detach()).abs().max().item()
    record('permutation_padding_and_masked_gradients',pe<1e-10 and ge<1e-8 and paderr<1e-10
           and torch.equal(bad,out.detach()) and padded[:,:,4:].abs().max()==0
           and all(g[~mask].abs().max()==0 for g in grad),
           permutation_abs=pe,permutation_gradient_abs=ge,padding_abs=paderr,
           masked_NaN_ignored=True,missing_input_gradients_zero=True)
    empty=torch.zeros_like(mask)
    all_missing=graph(torch.full_like(x,float('nan')),torch.full_like(z,float('nan')),empty,empty)
    record('all_missing',all_missing.abs().max()==0 and torch.isfinite(all_missing).all())
    return dict(passed=all(c['passed'] for c in checks.values()),device=device,formal_training=False,
                model='GNN', scope='original attention core with common six-input MLP and shared downstream LLM',
                source=str(PLAIN_SOURCE.relative_to(ROOT)),
                excluded_source_modules=['history_encoder','node_projection','time_embedding','decoder'],
                graph_trainable_parameters=sum(p.numel() for p in graph.parameters()),checks=checks)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--device',default='cuda:1')
    result=run(parser.parse_args().device)
    path=ROOT/'reports/f02/plain_baseline.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(path)
