from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
CURRENT = ROOT / "main.tex"


def extract_bibliography(text):
    section = text.split(r"\begin{thebibliography}{99}", 1)[1].split(
        r"\end{thebibliography}", 1
    )[0]
    matches = list(re.finditer(r"\\bibitem\{([^}]+)\}", section))
    entries = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section)
        entries[match.group(1)] = section[match.start():end].strip()
    return entries


old = CURRENT.read_text(encoding="utf-8")
entries = extract_bibliography(old)
entries["lin2022conditional"] = r"""\bibitem{lin2022conditional}
  H.~Lin, Z.~Gao, Y.~Xu, L.~Wu, L.~Li, and S.~Z.~Li.
  \newblock Conditional local convolution for spatio-temporal meteorological
    forecasting.
  \newblock \textit{Proc.\ AAAI Conf.\ Artif.\ Intell.},
    36(7):7470--7478, 2022.
  \newblock doi:10.1609/aaai.v36i7.20711."""

preamble = r"""\documentclass[fleqn,10pt]{wlscirep}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{float}
\usepackage{placeins}
\usepackage{multirow}
\usepackage{bm}

\newcommand{\R}{\mathbb{R}}
\graphicspath{{figs/}}

\title{Adaptive Graph Fusion Extensions to Conditional Local Convolution for Spherical Weather Forecasting}

\author[1,*]{Xinchen Geng}
\author[1]{Hung-Yu Lin}
\affil[1]{University of Southern California, Los Angeles, CA 90089, USA}
\affil[*]{gengx@usc.edu}

\keywords{Weather forecasting; Spatio-temporal graph neural networks; Adaptive graph learning; Spherical grids; Reproducibility}

\begin{abstract}
Conditional Local Convolution Recurrent Network (CLCRN) is an established
graph-recurrent model for meteorological forecasting. We examine whether a
compact adaptive graph-fusion (AGF) encoder can improve CLCRN by learning
non-local node dependencies before the original recurrent backbone. The
extension constructs a directed top-$k$ graph from trainable source and
destination node embeddings and combines graph-mixed and residual features
through a learned gate. Experiments use the WeatherBench preprocessing and
four tasks from the original CLCRN study: 2-m temperature, relative humidity,
10-m surface wind, and total cloud cover. We compare CLCRN-AGF with a
schedule-matched CLCRN control over five paired random seeds. The clearest
effect occurs for relative humidity, where CLCRN-AGF reduces mean absolute
error by 2.30\% and root mean squared error by 3.05\%; all five seeds improve,
although an exact two-sided sign-flip test is limited to $p=0.0625$ at
$n=5$. Effects on temperature, wind, and cloud cover are small or
metric-dependent. These results narrow the contribution of adaptive graph
fusion: it is useful for the humidity task but is not a uniformly beneficial
replacement for the original CLCRN encoder.
\end{abstract}

\begin{document}
\flushbottom
\maketitle
\thispagestyle{empty}
"""

body = r"""
\section*{Introduction}

Numerical weather prediction remains the operational standard because it
encodes atmospheric physics~\cite{bauer2015quiet}, while data-driven models
provide increasingly capable alternatives for selected forecasting tasks.
WeatherBench established a common ERA5-based benchmark for evaluating such
models~\cite{rasp2020weatherbench}. At the large-model end of the spectrum,
FourCastNet, Pangu-Weather, and GraphCast demonstrate the potential of neural
forecasting at global scale~\cite{pathak2022fourcastnet,bi2023accurate,lam2023learning}.
Smaller graph-recurrent systems remain relevant when retraining cost,
task-specific adaptation, and controlled experiments are important.

Conditional Local Convolution Recurrent Network (CLCRN) was introduced by
Lin et al.~\cite{lin2022conditional} for meteorological forecasting on an
unstructured spherical grid. Its Conditional Local Convolution (CLConv)
generates location-dependent kernels from spherical coordinates, geodesic
distance, and directional information. The published model therefore already
addresses the spatially shared-kernel limitation of conventional graph
convolutions. The present study does not claim CLCRN or CLConv as new.
Instead, it reproduces the public implementation and evaluates a specific
extension motivated by adaptive graph learning.

The extension, termed \textbf{CLCRN-AGF}, adds an adaptive graph-fusion
encoder before the original CLCRN recurrent backbone. Trainable source and
destination node embeddings define a directed similarity matrix. A sparse
top-$k$ neighbourhood is used to aggregate node features, and a learned gate
mixes the graph-aggregated representation with a residual projection. This
design tests a focused hypothesis: non-local learned dependencies may add
value beyond CLCRN's geographically conditioned local convolution,
particularly for variables whose dynamics are not fully described by local
proximity.

The study is framed as an extension and reproducibility analysis. We compare
CLCRN-AGF against a schedule-matched control that disables the complete AGF
module while retaining the same data, optimizer, training budget, and random
seeds. Published CLCRN values and an independent reproduction are reported
only as contextual references. The contributions are:

\begin{itemize}
\item an adaptive top-$k$ graph-fusion encoder attached to the established
CLCRN backbone;
\item a paired five-seed comparison against a schedule-matched control,
including confidence intervals and exact sign-flip tests;
\item a transparent separation between published reference values,
independent reproduction, and newly generated results; and
\item a negative as well as positive result: the extension consistently
helps relative humidity but gives small or metric-dependent changes for the
other tasks.
\end{itemize}

\section*{Related work}

\subsection*{Conditional local convolution}
CLCRN combines a sequence-to-sequence recurrent architecture with CLConv,
whose kernel is conditioned on the coordinates of each centre-neighbour
pair~\cite{lin2022conditional}. The original paper evaluated temperature,
relative humidity, surface wind, and cloud cover on a 2,048-node spherical
grid. CLConv and the CLCRN recurrent cell are prior work and serve as the
fixed backbone in the present study.

\subsection*{Adaptive graph forecasting}
Adaptive graph learning is commonly implemented with node embeddings,
attention scores, or learned adjacency matrices. AGCRN learns node-specific
patterns from adaptive embeddings~\cite{bai2020adaptive}; Graph WaveNet and
related models infer latent spatial dependencies alongside temporal
forecasting~\cite{wu2019graph,wu2020connecting}; and adaptive
spatio-temporal transformers revise fixed graph structure with attention
~\cite{feng2022adaptive,huang2023adaptive}. These methods motivate the
question addressed here: whether a non-local adaptive branch adds useful
information to a backbone that already contains geometry-conditioned local
kernels.

\subsection*{Compact weather models}
The experiments follow the original CLCRN task definition rather than the
full-state, medium-range setting used by recent global forecasters. Related
work has explored cost-efficient knowledge-oriented weather models,
spherical neural operators, and learned post-processing of numerical
forecasts~\cite{cheon2026costefficient,hu2025spherical,frnda2022ecmwf,alves2025spatiotemporal}.
Their datasets and objectives differ from the single-variable WeatherBench
tasks considered here, but they illustrate the continuing role of compact,
task-specific forecasting systems.

\section*{Results}

\subsection*{Primary paired comparison}
The primary comparison uses five matched random seeds
$\{2021,2022,2023,2024,2025\}$. The control is the original CLCRN backbone
trained with the same revised schedule as CLCRN-AGF. Table~\ref{tab:paired}
reports mean and standard deviation across seeds, the paired difference
(CLCRN-AGF minus control), a 95\% confidence interval for that difference,
and an exact two-sided sign-flip test.

\begin{table}[htbp]
\caption{Paired five-seed comparison of CLCRN-AGF and the schedule-matched
CLCRN control. Negative differences favour CLCRN-AGF. Confidence intervals
use the paired seed differences; exact two-sided sign-flip tests have a
minimum attainable $p$-value of 0.0625 for $n=5$.}
\label{tab:paired}
\centering
\small
\resizebox{\linewidth}{!}{%
\begin{tabular}{llrrrr}
\toprule
Task & Metric & Control ($\mu\pm\sigma$) & CLCRN-AGF ($\mu\pm\sigma$)
& Paired difference [95\% CI] & $p_{\mathrm{exact}}$ \\
\midrule
Temperature & MAE  & $1.229\pm0.057$ & $1.233\pm0.032$ & $+0.004$ [$-0.060,+0.068$] & 0.8125 \\
            & RMSE & $2.097\pm0.073$ & $2.020\pm0.047$ & $-0.076$ [$-0.171,+0.018$] & 0.1250 \\
Relative humidity & MAE  & $4.633\pm0.032$ & $4.527\pm0.014$ & $-0.107$ [$-0.153,-0.060$] & 0.0625 \\
                  & RMSE & $7.237\pm0.051$ & $7.016\pm0.041$ & $-0.221$ [$-0.251,-0.191$] & 0.0625 \\
Surface wind & MAE  & $1.244\pm0.005$ & $1.240\pm0.008$ & $-0.004$ [$-0.017,+0.009$] & 0.4375 \\
             & RMSE & $1.977\pm0.008$ & $1.973\pm0.017$ & $-0.004$ [$-0.030,+0.023$] & 0.6875 \\
Cloud cover & MAE  & $0.1490\pm0.0007$ & $0.1489\pm0.0008$ & $-0.0002$ [$-0.0013,+0.0009$] & 0.5625 \\
            & RMSE & $0.2449\pm0.0032$ & $0.2441\pm0.0017$ & $-0.0008$ [$-0.0047,+0.0031$] & 0.5625 \\
\bottomrule
\end{tabular}}
\end{table}

Relative humidity is the only task for which every paired seed favours
CLCRN-AGF for both metrics. Its mean MAE decreases by 2.30\% and RMSE by
3.05\%. With only five seed pairs, however, the exact two-sided test cannot
cross 0.05 even when all differences share the same sign. Temperature shows
a metric-dependent result: MAE is essentially unchanged, whereas mean RMSE
decreases by 3.64\% with an interval that includes zero. Wind and cloud-cover
changes are small relative to seed variability.

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{fig_primary_paired_comparison.pdf}
\caption{\textbf{Paired test MAE for the schedule-matched control and
CLCRN-AGF.} Each line joins results obtained with the same random seed;
diamonds show mean $\pm$ one standard deviation. The humidity improvement is
consistent across all five seeds, whereas the other tasks show overlapping or
mixed paired changes.}
\label{fig:paired}
\end{figure}

\subsection*{Contextual benchmarks}
Table~\ref{tab:context} distinguishes results generated in this study from
external published values. Persistence is evaluated directly on the current
test split. ``Published CLCRN'' is copied from Lin et al.~\cite{lin2022conditional}
after restoring the humidity and cloud-cover reporting scales used by the
current evaluation code. ``Reproduced CLCRN'' is a single independent run of
the public implementation. The two final rows are five-seed results generated
under the revised schedule.

\begin{table}[htbp]
\caption{Contextual test results. Published CLCRN values are external
references, not newly generated results. Reproduced CLCRN is a single run;
control and CLCRN-AGF are five-seed means. Lower is better.}
\label{tab:context}
\centering
\scriptsize
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
Method & \multicolumn{2}{c}{Temperature} & \multicolumn{2}{c}{Relative humidity}
& \multicolumn{2}{c}{Surface wind} & \multicolumn{2}{c}{Cloud cover}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
& MAE & RMSE & MAE & RMSE & MAE & RMSE & MAE & RMSE\\
\midrule
Persistence (current split) & 1.421 & 2.700 & 6.425 & 10.447 & 1.537 & 2.535 & 0.1620 & 0.2725\\
Published CLCRN~\cite{lin2022conditional} & 1.169 & 1.883 & 4.531 & 7.078 & 1.326 & 2.129 & 0.1491 & 0.2456\\
Reproduced CLCRN (one seed) & 1.192 & 2.246 & 4.677 & 7.309 & 1.384 & 2.266 & 0.1564 & 0.2574\\
Schedule-matched control (five seeds) & 1.229 & 2.097 & 4.633 & 7.237 & 1.244 & 1.977 & 0.1490 & 0.2449\\
CLCRN-AGF (five seeds) & 1.233 & 2.020 & 4.527 & 7.016 & 1.240 & 1.973 & 0.1489 & 0.2441\\
\bottomrule
\end{tabular}}
\end{table}

The contextual comparison shows why the extension should not be described as
uniformly superior. CLCRN-AGF improves on the published reference for
relative humidity, wind, and cloud cover, but not for temperature. Differences
from the external reference also combine architecture, software environment,
and training-protocol effects; the paired control in Table~\ref{tab:paired}
is therefore the primary causal comparison.

\subsection*{Computational cost}
The AGF encoder approximately doubles the number of trainable parameters and
increases inference memory because it constructs a dense node-similarity
matrix before top-$k$ selection. Table~\ref{tab:efficiency} reports direct
measurements on the same NVIDIA GeForce RTX 5090 Laptop GPU. The extension is
compact relative to global foundation models, but it is not cost-free and is
not asymptotically linear in the number of nodes.

\begin{table}[htbp]
\caption{Measured model size and inference cost. Timings are milliseconds per
test batch; memory is peak allocated GPU memory during inference.}
\label{tab:efficiency}
\centering
\small
\begin{tabular}{llrrr}
\toprule
Task & Variant & Parameters & Inference (ms) & Memory (MB)\\
\midrule
Temperature & CLCRN & 39,268 & 202.6 & 352.3\\
            & CLCRN-AGF & 80,892 & 211.9 & 634.5\\
Relative humidity & CLCRN & 39,268 & 206.7 & 353.1\\
                  & CLCRN-AGF & 80,892 & 212.4 & 634.5\\
Surface wind & CLCRN & 39,437 & 197.3 & 368.6\\
             & CLCRN-AGF & 81,061 & 207.4 & 644.6\\
Cloud cover & CLCRN & 39,268 & 202.0 & 353.1\\
            & CLCRN-AGF & 80,892 & 203.4 & 634.5\\
\bottomrule
\end{tabular}
\end{table}

\subsection*{Gate diagnostics}
Gate values are summarized only as model diagnostics; their marginal
distribution does not by itself establish a physical interpretation. The
single-seed distributions in Figure~\ref{fig:gate} differ across tasks, but
causal attribution would require interventions on the two branches or
independent meteorological labels.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.82\linewidth]{fig_gate_distribution.pdf}
\caption{\textbf{Distribution of mean gate activations for three tasks in a
representative trained model.} The plot describes how the learned mixture
varies across samples. It should not be interpreted as proof that either
branch corresponds to a specific physical process.}
\label{fig:gate}
\end{figure}

\section*{Discussion}

The paired experiment supports a narrow conclusion. Adding adaptive graph
fusion to CLCRN is beneficial for relative humidity under the tested setup:
the improvement is consistent across all five seeds and its paired confidence
interval excludes zero for both MAE and RMSE. The exact randomization test
remains limited by the small number of seed pairs. For temperature, wind, and
cloud cover, the evidence does not support a general claim of architectural
superiority.

This task dependence is plausible but should not be over-interpreted.
Relative humidity may benefit from dependencies that are not captured by the
fixed local neighbourhood alone. The current experiment does not demonstrate
that the learned edges correspond to teleconnections or moisture corridors.
Establishing such a claim would require spatial edge analysis, comparisons
with meteorological indices, and perturbation tests.

The study also separates optimization effects from architectural effects.
The revised schedule substantially improves wind and cloud-cover results even
without AGF, while the additional AGF gain is small. Reporting only the best
single run would obscure this distinction. The five-seed paired design is
therefore central to the present conclusions.

Several limitations remain. First, the study inherits the compact
single-variable setup of the original CLCRN paper rather than evaluating
full-state medium-range weather prediction. Second, the current baseline set
contains persistence, the published CLCRN reference, and a schedule-matched
control; modern graph and non-graph baselines still need to be retrained under
the identical data and optimization pipeline. Third, the top-$k$ graph is
obtained from a dense $N\times N$ similarity matrix, giving quadratic
intermediate cost. Fourth, individual full-budget ablations of graph mixing
and gated fusion are not yet available across all four tasks. These missing
comparisons should be completed before making stronger claims about either
component.

\section*{Conclusion}

This work evaluates an adaptive graph-fusion extension to the published
CLCRN weather-forecasting model. The revised manuscript distinguishes prior
CLCRN contributions, external published values, independent reproduction, and
new experiments. In paired five-seed tests, CLCRN-AGF consistently improves
relative-humidity forecasting, while changes for temperature, wind, and cloud
cover are small or metric-dependent. The results therefore support adaptive
graph fusion as a task-dependent extension rather than a universally better
replacement for the original CLCRN encoder.

\section*{Methods}

\subsection*{Data and preprocessing}
We use the same four WeatherBench-derived tasks and public preprocessing
pipeline as the original CLCRN study~\cite{lin2022conditional}: 2-m
temperature (K), relative humidity (percentage points), 10-m surface wind
components ($u$ and $v$, m\,s$^{-1}$), and total cloud cover (fraction).
Hourly ERA5 records from 1 January 2010 through 31 December 2018 are mapped to
a $32\times64$ latitude--longitude grid with $N=2{,}048$ nodes.

Each example contains 12 consecutive hourly observations and targets the next
12 hours. Starting windows are sampled once per day (step size 24), producing
3,286 examples. The split is chronological and unshuffled: 2,300 training
examples (70\%), 329 validation examples (10\%), and 657 test examples
(20\%). Input and target variables are standardized with means and
standard deviations computed from the training subset only; metrics are
calculated after inverse transformation.

\subsection*{CLCRN backbone}
The backbone is the public CLCRN implementation of Lin et
al.~\cite{lin2022conditional}. CLConv generates a local kernel from each
centre-neighbour coordinate pair and multiplies it by geodesic-distance and
orientation factors. The resulting operator replaces linear transforms in a
graph-convolutional GRU and is used in a sequence-to-sequence
encoder--decoder. We retain this backbone without claiming its architecture
as a contribution of the present study.

\subsection*{Adaptive graph-fusion encoder}
Let $\mathbf{F}\in\R^{B\times T\times N\times d}$ be the concatenation of
the projected weather feature and the learned node embedding. Trainable
source and destination embeddings
$\mathbf{E}_{s},\mathbf{E}_{t}\in\R^{N\times d_e}$ define
\begin{equation}
\mathbf{A}=\operatorname{softmax}(\mathbf{E}_{s}\mathbf{E}_{t}^{\top}),
\end{equation}
where softmax is applied row-wise. For each node, the implementation retains
the self-edge and the $k-1$ largest remaining entries, then renormalizes the
selected weights. With neighbour set $\mathcal{T}_k(i)$,
\begin{equation}
\mathbf{m}_{i}=\sum_{j\in\mathcal{T}_k(i)}\tilde A_{ij}\mathbf{F}_{j}.
\end{equation}
Graph-mixed and residual features are projected separately:
\begin{equation}
\mathbf{q}_{i}=\mathbf{W}_{m}\mathbf{m}_{i}+\mathbf{b}_{m},\qquad
\mathbf{r}_{i}=\mathbf{W}_{r}\mathbf{F}_{i}+\mathbf{b}_{r}.
\end{equation}
The gated output is
\begin{align}
\mathbf{g}_{i}&=\sigma\!\left(\mathbf{W}_{g}
[\mathbf{r}_{i}\Vert\mathbf{q}_{i}]+\mathbf{b}_{g}\right),\\
\mathbf{z}_{i}&=\mathbf{g}_{i}\odot\mathbf{q}_{i}
+(1-\mathbf{g}_{i})\odot\mathbf{r}_{i}.
\end{align}
$\mathbf{z}$ is concatenated with the standard CLCRN encoder inputs before
the unchanged recurrent encoder. The control sets the AGF encoder off, so it
removes both adaptive mixing and its gate.

\subsection*{Training configuration}
Both variants use two recurrent layers, 32 hidden units, an 8-dimensional
weather-feature embedding, an 8-dimensional CLConv embedding, one CLConv
propagation view, and curriculum learning in the autoregressive decoder.
CLCRN-AGF uses 10-dimensional adaptive node embeddings, an 8-dimensional AGF
output, and $k=4$ selected neighbours including the self-edge. Training uses
Adam with learning rate 0.01, batch size 32, gradient-norm clipping at 5,
and a MultiStepLR schedule with milestones
$\{10,20,30,40,50\}$ and multiplicative factor 0.95. Models train for at
most 100 epochs and the checkpoint with the lowest validation MAE is used for
test evaluation.

Experiments were run with Python 3.13, PyTorch 2.10, CUDA 12.8, and an NVIDIA
GeForce RTX 5090 Laptop GPU. Seeds 2021--2025 are applied to NumPy and all
PyTorch random-number generators.

\subsection*{Metrics and statistical analysis}
MAE and RMSE are computed over all 12 forecast steps, nodes, test examples,
and output channels after inverse scaling. Persistence repeats the final
input observation over the complete forecast horizon. The primary analysis
pairs CLCRN-AGF and control runs by seed. For each metric we report the
mean and sample standard deviation across seeds, the mean paired difference,
and a two-sided 95\% Student-$t$ confidence interval for the paired
differences. We additionally enumerate all $2^5$ sign flips to obtain an exact
two-sided randomization $p$-value. With five pairs, the smallest attainable
two-sided value is 0.0625, so these tests should be interpreted together with
effect sizes and consistency across seeds.

\subsection*{Computational complexity}
The original CLConv implementation materializes dense $N\times N$ kernel
operators, and AGF constructs a dense $N\times N$ similarity matrix before
top-$k$ selection. The implementation therefore has quadratic intermediate
memory and similarity-computation cost in $N$, despite sparse aggregation
after selection. Claims of linear scaling are not made.

\section*{Data availability}
The raw ERA5 variables are available through WeatherBench and the Copernicus
Climate Data Store. The preprocessing period, variables, temporal windows,
and chronological split are specified above. The exact preprocessed files
used for the experiments can be supplied to editors and reviewers subject to
the redistribution terms of the source data.

\section*{Code availability}
The study uses the public CLCRN implementation with the AGF module,
configuration files, and experiment scripts contained in the project
repository. A review archive containing exact configurations and per-seed
summaries will be supplied with submission, and a versioned public repository
will be archived before publication.
"""

tail = r"""
\section*{Funding}
This research received no specific grant from any funding agency in the
public, commercial, or not-for-profit sectors.

\section*{Author contributions}
X.G. conceived the extension, implemented the software, conducted the
experiments and analysis, prepared the visualizations, and wrote the original
draft. H.-Y.L. supervised the study and reviewed and edited the manuscript.
Both authors reviewed and approved the final manuscript.

\section*{Competing interests}
The authors declare no competing interests.

\section*{Generative AI and AI-assisted technologies disclosure}
During preparation of this manuscript, the authors used AI-assisted language
editing to improve readability and organization. The authors reviewed and
edited all generated content and take full responsibility for the final
manuscript.

\end{document}
"""

citation_order = []
for match in re.finditer(r"\\cite\{([^}]+)\}", body):
    for key in match.group(1).split(","):
        key = key.strip()
        if key and key not in citation_order:
            citation_order.append(key)

missing = [key for key in citation_order if key not in entries]
if missing:
    raise RuntimeError(f"Missing references: {missing}")

bibliography = "\n\n".join(entries[key] for key in citation_order)
document = (
    preamble
    + body
    + "\n\n\\FloatBarrier\n\\begin{thebibliography}{99}\n\n"
    + bibliography
    + "\n\n\\end{thebibliography}\n\n"
    + tail
)
CURRENT.write_text(document, encoding="utf-8", newline="\n")
print(f"Wrote {CURRENT} with {len(citation_order)} references")
