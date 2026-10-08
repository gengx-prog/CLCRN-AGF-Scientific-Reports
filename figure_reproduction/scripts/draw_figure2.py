"""Vector re-drawing of the original three-panel CLCRN-AGF Figure 2.

The composition and pastel colour language follow the original manuscript.
All computation labels and arrows describe the current released implementation.
No raster input, sampled data, external assets or generative images are used.
Run: python draw_figure2.py --output-dir figures
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon

DARK='#224751'
BLUE='#acd5e9'
PEACH='#efad92'
PURPLE='#c4b0d0'
GREEN='#bad8ae'
GOLD='#f5cc80'
TEAL='#13838a'
WHITE='#ffffff'

def main():
    parser=argparse.ArgumentParser()
    destination=parser.add_mutually_exclusive_group()
    destination.add_argument('--output-dir',type=Path,default=None)
    destination.add_argument('--output-root',type=Path,default=None,
                             help='Write figures/ and figure2_metadata.json below this root.')
    args=parser.parse_args()
    out=(args.output_root/'figures') if args.output_root else (args.output_dir or Path(__file__).parent/'figures')
    out.mkdir(parents=True,exist_ok=True)
    fonts={f.name for f in font_manager.fontManager.ttflist}
    family='Comic Sans MS' if 'Comic Sans MS' in fonts else 'DejaVu Sans'
    plt.rcParams.update({'font.family':family,'font.size':11.5,'pdf.fonttype':42,
                         'ps.fonttype':42,'svg.fonttype':'path','mathtext.fontset':'dejavusans'})
    fig=plt.figure(figsize=(17.95,10.08),facecolor='white')
    ax=fig.add_axes([.008,.012,.984,.976]); ax.set(xlim=(0,18),ylim=(0,10)); ax.axis('off')
    texts=[]
    def text(x,y,s,size=11.5,ha='center',va='center',weight='normal',rotation=0,color=DARK,**kw):
        t=ax.text(x,y,s,fontsize=size,ha=ha,va=va,fontweight=weight,rotation=rotation,
                  color=color,zorder=10,linespacing=1.2,**kw); texts.append(t); return t
    def box(x,y,w,h,c='white',r=.11,lw=1.4):
        p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.012,rounding_size={r}',
                        linewidth=lw,edgecolor=DARK,facecolor=c,zorder=2); ax.add_patch(p); return p
    def line(points,lw=1.3,color=DARK,style='-',z=3):
        ax.plot(*zip(*points),lw=lw,color=color,ls=style,zorder=z,solid_capstyle='round')
    def arrow(a,b,lw=1.4,head=12,color=DARK,style='-',connection='arc3,rad=0'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=head,lw=lw,color=color,
                                    linestyle=style,connectionstyle=connection,zorder=7))
    def route(points,lw=1.3,color=DARK,style='-'):
        line(points[:-1],lw=lw,color=color,style=style)
        arrow(points[-2],points[-1],lw=lw,color=color,style=style)
    def node(x,y,r=.085,c=TEAL):
        ax.add_patch(Circle((x,y),r,facecolor=c,edgecolor=DARK,lw=1.05,zorder=6))
    def matrix(x,y,w,h,n=6,c=BLUE,sparse=False,labels=False):
        # Purely illustrative cells; these are not a measured adjacency.
        for i in range(n):
            for j in range(n):
                active=(j in {(i+d)%n for d in (0,1,3,5)}) if sparse else True
                v=(.25+.70*((3*i+5*j)%13)/12) if active else .02
                rgb=np.array(matplotlib.colors.to_rgb(c)); fc=tuple(1-(1-rgb)*v)
                ax.add_patch(Rectangle((x+j*w/n,y+(n-1-i)*h/n),w/n,h/n,
                                      facecolor=fc,edgecolor='#61777a',lw=.35,zorder=4))
        ax.add_patch(Rectangle((x,y),w,h,facecolor='none',edgecolor=DARK,lw=1.1,zorder=5))
    def trace(x0,x1,y,amp=.28,shade=None):
        if shade:
            ax.add_patch(Rectangle((shade,y-amp-.07),x1-shade,2*amp+.13,fc='#d9e9fa',ec='none',zorder=1))
        xx=np.linspace(x0,x1,27)
        for phase,c in [(.0,'#43a591'),(.72,'#477fad'),(1.35,'#8d60af')]:
            yy=y+amp*np.sin(np.linspace(0,3*np.pi,27)+phase)
            ax.plot(xx,yy,color=c,lw=1.1,marker='o',ms=2.8,mew=0,zorder=4)
    def up(x,y0,y1):
        ax.add_patch(FancyArrowPatch((x,y0),(x,y1),arrowstyle='simple',mutation_scale=15,
                                    lw=1.1,edgecolor=DARK,facecolor='white',zorder=4))

    # The panel geometry and upward main pipeline reproduce the original figure.
    box(.04,.25,5.48,9.6,r=.38,lw=1.75)
    box(5.70,4.61,12.24,5.24,r=.36,lw=1.75)
    box(5.70,.25,12.24,4.06,r=.36,lw=1.75)

    # (a) End-to-end forecasting pipeline.
    text(.31,8.75,'Output',size=14,weight='bold',rotation=90)
    trace(.73,5.16,9.02,.33,shade=3.0)
    text(2.9,8.48,r'$\hat{x}_{t+1:t+12}\in\mathbb{R}^{12\times N\times C}$',size=13)
    text(2.87,9.57,'12-hour forecast',size=13.2,weight='bold')
    up(2.88,8.02,8.23)
    box(.62,7.30,4.53,.67,PEACH)
    text(2.88,7.75,'Autoregressive CLConv-GRU decoder',size=12.0,weight='bold')
    text(2.88,7.47,'Scheduled sampling during training only',size=10.3)
    up(2.88,6.78,7.21)
    box(.62,5.78,4.53,.94,PURPLE)
    text(2.88,6.48,'Stacked CLConv-GRU encoder',size=12.3,weight='bold')
    text(.89,6.05,r'$L=2$',size=11.1,ha='left')
    # Miniature recurrent-cell drawing retains the characteristic old detail.
    box(2.02,5.91,2.91,.32,c='#d9cde0',r=.07,lw=.85)
    for x,label in [(2.47,'r'),(3.39,'u'),(4.31,'c')]:
        box(x-.18,5.955,.36,.23,'#ece4ef',r=.035,lw=.8); text(x,6.07,label,size=8.5)
    arrow((2.69,6.07),(3.12,6.07),head=7,lw=.8)
    arrow((3.60,6.07),(4.04,6.07),head=7,lw=.8)
    route([(2.11,6.22),(2.11,6.36),(4.74,6.36),(4.74,6.18)],lw=.8)
    up(2.88,5.16,5.68)
    box(.62,4.47,4.53,.64,GREEN)
    text(2.88,4.89,'Retain input and concatenate',size=12.1,weight='bold')
    text(2.88,4.62,r'$[\,F\;\Vert\;\bar{x}\;\Vert\;z\,]$',size=12.5)
    up(2.88,3.84,4.36)
    box(.62,2.91,4.53,.88,BLUE)
    text(2.88,3.54,'Adaptive graph fusion (AGF)',size=12.4,weight='bold')
    matrix(1.83,3.17,.28,.24,n=4,c=BLUE,sparse=True)
    text(2.35,3.29,r'$\widetilde{A}$',size=10.5)
    arrow((2.62,3.29),(3.05,3.29),head=8,lw=.9)
    for j,c in enumerate([BLUE,GOLD,GREEN]):
        ax.add_patch(Rectangle((3.19+j*.11,3.17),.1,.24,fc=c,ec=DARK,lw=.5,zorder=4))
    text(3.82,3.29,r'$z$',size=11.5)
    text(2.89,3.035,r'Pre-CLConv feature $z$; detail in (c)',size=10.2)
    up(2.88,2.37,2.80)
    box(.62,1.63,4.53,.69,PEACH)
    text(2.88,2.08,'Weather projection + node embedding',size=11.5,weight='bold')
    text(2.88,1.80,r'$F=[\,\phi_x(\bar{x})\;\Vert\;e_i\,]$',size=12.1)
    up(2.88,1.26,1.54)
    trace(.72,5.16,.86,.24)
    text(.31,.87,'Input',size=14,weight='bold',rotation=90)
    text(2.88,.40,r'$\bar{x}_{t-11:t}$'+': 12 hourly inputs; '+r'$N=2,048$',size=11.2)
    # Bypass is essential: z does not replace the original encoder features.
    route([(.62,1.93),(.43,1.93),(.43,4.80),(.62,4.80)],style='--',lw=1.05)
    route([(5.16,.85),(5.33,.85),(5.33,4.79),(5.16,4.79)],style='--',lw=1.05)
    text(5.33,2.33,r'$\bar{x}$',size=10.5,ha='right')

    # (b) Local geometry, kernel generation, and ONE propagation view.
    for x,s in [(7.60,'STAGE 1'),(11.08,'STAGE 2'),(15.58,'STAGE 3')]:
        text(x,9.60,s,size=12.4,weight='bold')
    text(7.61,9.20,'Local geographic coordinates',size=11.1)
    center=(7.08,7.28)
    angles=np.array([-80,-45,-9,24,53,86,122,167])
    lengths=np.array([1.5,1.8,1.95,1.76,1.72,1.69,1.54,1.30])
    for j,(ang,ll) in enumerate(zip(angles,lengths)):
        theta=np.deg2rad(ang); xx=center[0]+ll*.76*np.cos(theta); yy=center[1]+ll*.77*np.sin(theta)
        arrow((xx,yy),center,lw=.95,head=8,color='#405359');node(xx,yy,.095)
        if j in [0,2,4,6]:
            text(xx+.11,yy+.17,rf'$v_{{i,{j+1}}}$',size=12,ha='left')
    node(*center,r=.19,c='#127a80'); text(6.74,7.10,r'$v_i$',size=13)
    text(6.61,8.67,'K = 25',size=12,weight='bold')
    text(7.50,5.55,'Fixed geographic neighbourhood',size=10.6)
    text(7.51,5.17,'(schematic node positions)',size=9.1)
    box(9.39,6.18,3.31,2.80,GOLD,r=.28,lw=1.4)
    text(11.045,8.65,'MLP kernel generator',size=12.6,weight='bold')
    text(11.045,8.24,r'$\ell_{ij}=\mathrm{MLP}(\mathrm{loc}_{ij})$',size=12.4)
    # Small network layers keep the old illustration's fine structure.
    for y,s in [(7.75,'Hidden widths 10, 8, 6'),(7.26,'Batch norm / tanh'),(6.77,'Scalar output + ReLU')]:
        box(9.95,y-.13,2.19,.29,'#ffe7b7',r=.035,lw=.7); text(11.045,y+.015,s,size=8.7)
    arrow((11.045,8.02),(11.045,7.91),head=7,lw=.8)
    arrow((11.045,7.58),(11.045,7.42),head=7,lw=.8)
    arrow((11.045,7.09),(11.045,6.94),head=7,lw=.8)
    text(11.045,6.40,'Non-negative local weights',size=9.4)
    route([(8.76,7.27),(9.09,7.27),(9.09,7.78),(9.38,7.78)])
    box(9.10,4.91,2.15,.69,BLUE,r=.07,lw=1.0)
    text(10.175,5.39,'Geodesic decay',size=9.8,weight='bold')
    text(10.175,5.10,r'$d_{ij}=\exp(-\alpha_{ij}s_{ij})$',size=10.4)
    box(11.51,4.91,2.15,.69,GREEN,r=.07,lw=1.0)
    text(12.585,5.39,'Angular weighting',size=9.5,weight='bold')
    text(12.585,5.10,r'$a_{ij}$',size=12.2)
    # A multiplication circle combines all three actual kernel factors.
    node(13.31,7.28,.135,c='white');text(13.31,7.29,r'$\times$',size=15)
    route([(12.72,7.28),(13.16,7.28)])
    route([(10.175,5.62),(10.175,5.87),(13.31,5.87),(13.31,7.13)],lw=1.1)
    route([(12.585,5.62),(12.585,5.87),(13.31,5.87),(13.31,7.13)],lw=1.1)
    text(13.38,7.73,r'$K_{ij}=\ell_{ij}\,d_{ij}\,a_{ij}$',size=11.0)
    box(14.10,6.14,2.40,2.67,PURPLE,r=.23,lw=1.4)
    text(15.30,8.53,'Local aggregation',size=11.9,weight='bold')
    matrix(14.78,7.27,1.02,.91,n=7,c='#8c73a6',sparse=True)
    text(15.30,6.93,r'$K H$',size=17)
    text(15.30,6.52,'One kernel view',size=10.9)
    arrow((13.46,7.28),(14.05,7.28))
    # Slanted learnable projection like the original W0 icon.
    ax.add_patch(Polygon([[16.85,6.98],[17.53,6.98],[17.65,7.55],[16.97,7.55]],
                         fc=GOLD,ec=DARK,lw=1.2,zorder=3))
    text(17.24,7.27,r'$W$',size=18)
    arrow((16.53,7.28),(16.84,7.28))
    arrow((17.65,7.28),(17.82,7.28))
    text(17.23,6.63,'Recurrent\ngates',size=10.4)
    text(15.42,5.51,'Used inside both encoder',size=10.7)
    text(15.42,5.18,'and decoder CLConv-GRU cells',size=10.7)

    # (c) Adjacency construction and feature-conditioned fusion of the same F.
    # Upper band follows the original graph/matrix visual language.
    box(6.04,3.00,11.53,1.01,c='#fff7e7',r=.14,lw=1.05)
    text(6.89,3.78,'Learned node',size=9.6,weight='bold')
    text(6.89,3.57,'embeddings',size=9.6,weight='bold')
    for xx,label in [(6.49,r'$E_s$'),(7.15,r'$E_t$')]:
        matrix(xx,3.08,.39,.36,n=4,c=GOLD);text(xx+.20,3.28,label,size=11.6)
    arrow((7.64,3.48),(7.93,3.48))
    text(8.93,3.65,r'$A=\mathrm{softmax}_{\rm row}$',size=11.7)
    text(8.93,3.24,r'$(E_s E_t^{\mathsf{T}})$',size=12.4)
    matrix(10.02,3.13,.70,.70,n=7,c=GOLD)
    arrow((10.78,3.48),(11.19,3.48))
    text(12.48,3.67,'Keep self + top-3 others',size=11.1,weight='bold')
    text(12.48,3.29,'Renormalize each row',size=10.9)
    arrow((13.79,3.48),(14.19,3.48))
    matrix(14.24,3.13,.70,.70,n=7,c=BLUE,sparse=True)
    text(15.19,3.47,r'$\widetilde{A}$',size=17)
    text(16.33,3.61,'k = 4',size=12.2,weight='bold')
    text(16.33,3.26,'including self',size=9.8)
    box(6.05,1.36,.76,.70,BLUE,r=.10,lw=1.2)
    text(6.43,1.71,r'$F$',size=17)
    text(6.45,1.09,'Same input',size=8.8)
    box(7.47,1.97,3.71,.65,GOLD,r=.12,lw=1.2)
    text(9.325,2.42,'Graph aggregation + projection',size=10.5,weight='bold')
    text(9.325,2.14,r'$q=\mathrm{Linear}_{m}(\widetilde{A}F)$',size=12.4)
    box(7.47,.72,3.71,.65,BLUE,r=.12,lw=1.2)
    text(9.325,1.16,'Node-wise residual projection',size=10.5,weight='bold')
    text(9.325,.89,r'$r=\mathrm{Linear}_{r}(F)$',size=12.4)
    route([(6.84,1.84),(7.11,1.84),(7.11,2.28),(7.44,2.28)])
    route([(6.84,1.56),(7.11,1.56),(7.11,1.05),(7.44,1.05)])
    route([(14.60,3.10),(14.60,2.81),(9.28,2.81),(9.28,2.65)],lw=1.05)
    box(11.91,1.38,2.27,.71,GREEN,r=.11,lw=1.2)
    text(13.045,1.89,'Feature gate',size=10.7,weight='bold')
    text(13.045,1.58,r'$g=\sigma(\mathrm{Linear}_{g}([r\Vert q]))$',size=10.2)
    route([(11.22,2.28),(11.56,2.28),(11.56,1.91),(11.88,1.91)],lw=1.15)
    route([(11.22,1.05),(11.56,1.05),(11.56,1.56),(11.88,1.56)],lw=1.15)
    # The gate produces weights; q and r also enter the final weighted sum.
    route([(11.20,2.49),(11.41,2.49),(11.41,2.70),(16.22,2.70),(16.22,2.11)],lw=1.05)
    route([(11.20,.90),(11.41,.90),(11.41,.54),(16.22,.54),(16.22,1.36)],lw=1.05)
    text(15.18,2.69,r'$q$',size=10.4,va='bottom')
    text(15.18,.57,r'$r$',size=10.4,va='bottom')
    arrow((14.22,1.73),(14.89,1.73),lw=1.3);text(14.56,1.90,r'$g$',size=10.3)
    box(14.93,1.38,2.52,.71,PEACH,r=.11,lw=1.2)
    text(16.19,1.89,'Gated fusion',size=11.0,weight='bold')
    text(16.19,1.58,r'$z=g\odot q+(1-g)\odot r$',size=11.4)
    text(12.93,1.06,'Input-dependent gate',size=9.1)
    text(6.06,.56,'Graph is fixed for a trained checkpoint.',size=8.8,ha='left')
    # Panel labels sit outside the borders just as in the earlier paper figure.
    text(2.8,.105,'(a) Overall encoder–decoder pipeline',size=12.2)
    text(11.82,4.46,'(b) Conditional local convolution (CLConv)',size=12.2)
    text(11.82,.105,'(c) Adaptive graph construction and gated feature fusion',size=12.2)

    fig.canvas.draw()
    # Assert that visible text remains within the figure, not clipped at export.
    renderer=fig.canvas.get_renderer(); outside=[]
    for t in texts:
        bb=t.get_window_extent(renderer)
        if bb.x0<0 or bb.y0<0 or bb.x1>fig.bbox.width or bb.y1>fig.bbox.height:
            outside.append(t.get_text())
    if outside: raise RuntimeError(f'Text outside canvas: {outside}')
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'fig2_architecture.{ext}',dpi=250,facecolor='white',metadata={'Creator':'Matplotlib; draw_figure2.py'} if ext=='pdf' else None)
    plt.close(fig)
    meta={'style':'Original three-panel composition, upward pipeline, round pastel boxes, dotted node graphs, matrix grids, recurrent cell and projection icons, and original Comic Sans typography.',
          'font':family,'native_vector':True,'illustrative_only':'All traces, graph node positions and matrix cells are schematic, not experimental data.',
          'corrections':['AGF is applied before CLConv-GRU','Original weather and embeddings bypass AGF and are concatenated with z','Learned adjacency uses softmax(Es Et.T), not an antisymmetric matrix','Self plus three other neighbours gives total k=4','Both projected branches originate from the same F','Gate weights graph q as g and residual r as 1-g','One actual kernel view, not multi-hop propagation','No fictitious binary/distance/adaptive triple-graph fusion','Teacher forcing only during training'],
          'code_alignment':{'node_num':2048,'seq_len':12,'horizon':12,'k_neighbors':25,'max_view':1,'layer_num':2,'asttn_topk':4},
          'caption':'CLCRN-AGF architecture retaining the original three-panel visual design. (a) Standardized weather inputs are projected and concatenated with learned node embeddings to form F. AGF acts on F before the recurrent backbone; its output z is concatenated with F and the standardized weather input. A two-layer CLConv-GRU encoder and autoregressive decoder predict the next 12 hours. Scheduled sampling is used only during training. (b) The local kernel combines learned coordinate-dependent weights, geodesic decay and angular weighting over 25 geographic neighbours. These experiments use a single kernel view in the recurrent cells. (c) Row-wise softmax of learned source and destination embeddings defines an input-independent adjacency for each checkpoint. Self plus the three strongest other neighbours are retained and renormalized. Graph-aggregated and residual projections q and r use the same pre-CLConv F; their input-dependent gate forms z=g*q+(1-g)*r. The control omits z and retains the original weather and node embeddings. Traces, node positions and matrix cells are illustrative.',
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('fig2_architecture.*'))}}
    (out.parent/'figure2_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'output_dir':str(out),'font':family,'outside_text':outside,'files':list(meta['outputs'])}))

if __name__=='__main__':main()
