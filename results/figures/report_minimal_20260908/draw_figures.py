from pathlib import Path
import sys, json, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from PIL import Image
ROOT=Path('/home/js_cn/sensing')
OUT=ROOT/'figures/report_minimal_20260908'
sys.path.insert(0,str(OUT/'skill'))
from setup_style import setup_style
from visual_qa import audit_layout
setup_style(journal='general',lang='en',constrained_layout=False)
from matplotlib import font_manager
for font in Path('/usr/share/fonts').rglob('*wqy*ttc'): font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':'WenQuanYi Micro Hei','font.size':13,'axes.titlesize':16,'axes.labelsize':13,'xtick.labelsize':12,'ytick.labelsize':13,'svg.fonttype':'path','pdf.fonttype':42,'figure.facecolor':'white','axes.facecolor':'white'})
B='#0072B2'; O='#D55E00'; K='#26333B'; G='#66747D'
FIGS=[]
def save(fig,name):
    issues=audit_layout(fig)
    print(name,issues)
    if any(i[0]=='FAIL' for i in issues): raise RuntimeError(issues)
    fig.savefig(OUT/(name+'.png'),dpi=300,facecolor='white')
    fig.savefig(OUT/(name+'_preview.png'),dpi=115,facecolor='white')
    Image.open(OUT/(name+'_preview.png')).convert('L').save(OUT/(name+'_gray.png'))
    if '--final' in sys.argv:
        for ext in ['svg','pdf']: fig.savefig(OUT/(name+'.'+ext),facecolor='white')
    FIGS.append(name)
def canvas(w=14,h=6):
    fig,ax=plt.subplots(figsize=(w,h));fig.subplots_adjust(0.015,0.02,.985,.98)
    ax.set(xlim=(0,14),ylim=(0,6));ax.axis('off');return fig,ax
def box(ax,x,y,w,h,t,c=B,fs=13):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.09',facecolor='white',edgecolor=c,lw=1.5))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,color=K)
def arr(ax,x,y,u,v,c=G):
    ax.annotate('',xy=(u,v),xytext=(x,y),arrowprops=dict(arrowstyle='-|>',color=c,lw=1.5,shrinkA=3,shrinkB=3))

# Schematic 1: actual forward path, including graph trajectory residual.
f,a=canvas();a.text(.2,5.65,'完整预测流程：QGNN 在哪里起作用？',fontsize=20,weight='bold',color=K)
steps=[('含噪雷达量测',.15,1.7),('关联图\n恢复目标历史',2.25,1.7),('20 帧位置、速度\n构建目标交互图',4.35,2.0),('第一层经典 GNN\n第二层量子消息',6.75,2.1)]
for t,x,w in steps:box(a,x,3.85,w,.9,t)
for x,u in [(1.85,2.25),(3.95,4.35),(6.35,6.75)]:arr(a,x,4.3,u,4.3)
box(a,10,4.25,3.2,.75,'图预测头：基础未来轨迹',G)
box(a,9.35,2.95,3.9,.8,'图节点特征＋历史运动软嵌入',B)
arr(a,8.85,4.4,10,4.6);arr(a,8.85,4,9.35,3.45)
box(a,9.35,1.65,3.9,.8,'GPT-2＋LoRA → 坐标修正量',B)
arr(a,11.3,2.95,11.3,2.45)
box(a,9.35,.25,3.9,.75,'相加 → 未来 20 帧坐标',O)
arr(a,11.3,1.65,11.3,1);a.plot([13.2,13.65,13.65],[4.6,4.6,.63],color=G,lw=1.3);arr(a,13.65,.63,13.25,.63)
a.text(.3,2.7,'图网络：判断目标之间如何影响\n\n量子核心：决定关注谁、消息放大或衰减多少\n\nLLM：结合历史和图特征，修正基础预测',fontsize=15,va='top',linespacing=1.5,color=K)
a.text(.3,.35,'已知身份测试从目标历史开始；关联图用于未知身份完整链路。',fontsize=11,color=G)
save(f,'01_整体流程')

# Schematic 2: one complete circuit layer, all six CNOT gates sequential.
f,a=canvas(15,7);a.text(.15,5.73,'QGNN 消息核心：一条交互边的计算过程',fontsize=20,weight='bold',color=K)
for x,w,t in [(.2,3.1,'两端节点 128＋128 维\n＋物理边 7 维'),(3.8,3.5,'LayerNorm → Linear → π tanh\n得到 6 个输入角度'),(7.8,2.4,'6 比特电路\n下方单层重复 3 次'),(10.8,3.,'测量 6 个 Z＋6 个 ZZ\n得到 12 个期望值')]:box(a,x,4.75,w,.67,t,fs=11)
for x,u in [(3.3,3.8),(7.3,7.8),(10.2,10.8)]:arr(a,x,5.08,u,5.08)
ys=[4.02-i*.43 for i in range(6)]
a.text(.25,4.44,'展开一层',color=B,weight='bold',fontsize=12)
for q,y in enumerate(ys):
    a.text(.23,y,f'q{q}  |0>' if q==0 else f'q{q}',va='center',fontsize=11)
    a.plot([1,10.05],[y,y],color='#9DA7AC',lw=1,zorder=0)
    box(a,1.3,y-.16,1.05,.32,'RY(θ)',fs=11)
    box(a,2.7,y-.16,2.15,.32,'RZ(φ) RY(β) RZ(ω)',c=O,fs=10)
for q in range(6):
    x=5.4+q*.76;target=(q+1)%6
    a.plot([x,x],[ys[q],ys[target]],color=K,lw=1.3)
    a.add_patch(Circle((x,ys[q]),.038,color=K))
    a.add_patch(Circle((x,ys[target]),.087,facecolor='white',edgecolor=K,lw=1.3))
    a.plot([x-.067,x+.067],[ys[target],ys[target]],color=K,lw=1)
    a.plot([x,x],[ys[target]-.067,ys[target]+.067],color=K,lw=1)
a.text(1.82,1.46,'写入边数据',ha='center',fontsize=12,color=B)
a.text(3.77,1.46,'可训练旋转',ha='center',fontsize=12,color=O)
a.text(7.32,1.46,'依次执行的环形 CNOT（含 q5→q0）',ha='center',fontsize=11,color=K)
a.text(10.4,4.1,'12 维测量结果',fontsize=13,color=B,weight='bold')
a.text(10.4,3.56,'归一化＋两个线性读出',fontsize=11)
a.text(10.4,2.98,'softmax → 4 头注意力 α',fontsize=11)
a.text(10.4,2.4,'2 sigmoid → 4 头门控 g',fontsize=11)
a.text(10.4,1.83,'g∈(0, 2)：可衰减，也可放大',fontsize=10,color=G)
a.text(.3,.88,'每条边的消息 = 注意力 α × 门控 g × 128 维经典消息；对邻居求和，再更新节点。',fontsize=14,color=K)
a.text(.3,.36,'初态为六个 |0>；3 层共享输入 θ、各有训练参数。共 54 个旋转参数；ZZ 衡量相邻比特的相关性。',fontsize=11,color=G)
save(f,'02_量子电路与消息机制')

# Results use source JSON directly. Aggregate estimates, not training replicates.
base=ROOT/'diagnostics/core_message_seed2026_v2'
main=json.loads((base/'test_fixed_v1/paired_block_analysis.json').read_text())
abpath=next(p for p in base.rglob('paired_block_analysis.json') if 'mechanism' in str(p))
abl=json.loads(abpath.read_text());print('ablation source',abpath,abl.keys())
rows=[]
for group,src in [('test',main),('ablation',abl)]:
    for name,d in src['comparisons'].items():
        if 'llm' not in name: continue
        key='reduction_percent' if group=='test' else 'ablation_error_increase_percent'
        for metric in ['ade_m','fde_m']:
            lo,hi=d['block_bootstrap_95_percentile_ci'][metric]
            rows.append(dict(group=group,comparison=name,metric=metric,estimate=d[key][metric],low=lo,high=hi))
with (OUT/'plot_data.csv').open('w') as out:
    w=csv.DictWriter(out,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def forest(items,name,title,xlabel,xlim,caption):
    fig,ax=plt.subplots(figsize=(10,4.5));fig.subplots_adjust(left=.28,right=.96,top=.82,bottom=.25)
    for j,(lab,d,key) in enumerate(items):
        for metric,delta,c,m in [('ade_m',.12,B,'o'),('fde_m',-.12,O,'s')]:
            val=d[key][metric];lo,hi=d['block_bootstrap_95_percentile_ci'][metric];y=j+delta
            ax.errorbar(val,y,xerr=[[val-lo],[hi-val]],fmt=m,color=c,capsize=4,markersize=7,lw=1.7,label=metric[:3].upper() if j==0 else None)
            ax.text(hi+.06,y,f'{val:.2f}%',va='center',color=c,fontsize=11)
    ax.set_yticks(range(len(items)),[i[0] for i in items]);ax.invert_yaxis();ax.set_ylim(len(items)-.55,-.5)
    ax.set_xlim(*xlim);ax.axvline(0,color=G,ls='--',lw=1);ax.set_xlabel(xlabel,labelpad=9)
    ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,pad=12);ax.grid(axis='x',alpha=.17)
    ax.legend(loc='lower right',bbox_to_anchor=(1,1.03),ncol=2,frameon=False)
    fig.text(.04,.94,title,fontsize=18,weight='bold',color=K)
    fig.text(.04,.04,caption,fontsize=10,color=G)
    save(fig,name)
forest([('相对原始 GNN＋LLM',main['comparisons']['llm_vs_plain'],'reduction_percent'),('相对同接口经典核心＋LLM',main['comparisons']['llm_vs_classical'],'reduction_percent')],'03_测试集对比','测试集：量子方案的误差降低了多少？','误差相对降低（%）；正值表示 QGNN 更好',(-.6,2.8),'种子 2026；1,200 场景，9,070 轨迹，14 时间区块；10,000 次配对区块 bootstrap 的 95% CI。\n区间反映固定模型的测试区块变化，不代表多训练种子的波动。保留版本：PyTorch 解析态矢量模拟。')
labels={'no_entanglement':'去掉纠缠','depth1':'电路深度 3→1','no_reupload':'关闭数据重上传'}
items=[]
for key,d in abl['comparisons'].items():
    if key.endswith('_llm'):
        label=next((v for k,v in labels.items() if k in key),key)
        items.append((label,d,'ablation_error_increase_percent'))
forest(items,'04_电路消融','开发集：移除电路部件后，误差如何变化？','误差相对上升（%）；正值表示移除后变差',(0,5.7),f"种子 2026；991 场景，{abl.get('unique_blocks','')} 时间区块；10,000 次配对区块 bootstrap 的 95% CI。\n只使用开发集；完整电路为 6 比特、3 层、环形 CNOT、数据重上传。区间不包含重新训练的不确定性。")
print('completed',FIGS)
