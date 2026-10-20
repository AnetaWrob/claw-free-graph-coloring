"""Statistical analysis: LaTeX tables and PDF figures for the paper.

    python analysis.py [--backend highs]

Reads data/results/set_{A,B,C}_<backend>.csv, writes paper/tables/*.tex,
paper/figures/*.pdf and data/results/summary_<backend>.txt.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="highs")
args = ap.parse_args()
BE = args.backend
RES, TAB, FIG = "data/results", "paper/tables", "paper/figures"
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

A = pd.read_csv(f"{RES}/set_A_{BE}.csv")
B = pd.read_csv(f"{RES}/set_B_{BE}.csv")
C = pd.read_csv(f"{RES}/set_C_{BE}.csv")
ALL = pd.concat([A, B, C], ignore_index=True)
log = open(f"{RES}/summary_{BE}.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    log.write(s + "\n")


def write(name, body):
    with open(f"{TAB}/{name}.tex", "w") as fh:
        fh.write(body)


def ms(x, digits=1):
    """seconds -> milliseconds string"""
    if pd.isna(x):
        return "--"
    v = 1000.0 * x
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.{digits}f}"


def gmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


NAMES = {"er": "Erd\\H{o}s--R\\'enyi", "regular": "regular", "bipartite": "bipartite",
         "tree": "trees", "subdivided": "subdivided regular",
         "co-bipartite": "co-bipartite", "co-triangle-free": "co-triangle-free",
         "enumeration": "enumeration"}
PLAIN = {"er": "Erdős–Rényi", "regular": "regular", "bipartite": "bipartite",
         "tree": "trees", "subdivided": "subdivided regular",
         "co-bipartite": "co-bipartite", "co-triangle-free": "co-triangle-free"}
ORDER_A = ["er", "bipartite", "tree", "regular", "subdivided"]
ORDER_B = ["co-bipartite", "co-triangle-free"]

# ------------------------------------------------------------ consistency
say("== consistency ==")
for name, D in (("A", A), ("B", B), ("C", C)):
    say(name, "graphs:", len(D), "| inconsistent:", int((D.consistent == 0).sum()),
        "| line graphs:", int(D.is_line_graph.sum()))
say("C counts by n:", C.groupby("n").size().to_dict())
say("C line graphs by n:", C.groupby("n").is_line_graph.sum().to_dict())
say("C gap distribution:", pd.crosstab(C.family, C.chi_minus_omega).to_dict())
say("C class:", C[C.is_line_graph == 1].root_class.value_counts().to_dict())
say("B: reference chi available:", int(B.chi_reference.notna().sum()),
    "| dsatur solved:", int((B.dsatur_status == "optimal").sum()),
    "| vedge solved:", int((B.vedge_status == "optimal").sum()))
say("B agreement dsatur==ref on solved:",
    bool((B[B.dsatur_status == "optimal"].dsatur_chi == B[B.dsatur_status == "optimal"].chi_reference).all()),
    "| vedge==ref on solved:",
    bool((B[B.vedge_status == "optimal"].vedge_chi == B[B.vedge_status == "optimal"].chi_reference).all()))

# ------------------------------------------------------------ Table 1
rows = []
for label, D, order in (("A", A, ORDER_A), ("B", B, ORDER_B)):
    for src in order:
        d = D[D.source == src]
        rows.append((label, NAMES[src], d))
for fam in ("line", "non-line claw-free"):
    rows.append(("C", fam.replace("non-line claw-free", "non-line"), C[C.family == fam]))
body = ("\\begin{tabular}{llrrrrr}\n\\toprule\nSet & Family & Graphs & $n$ & $m$ & "
        "Mean density & Line graphs\\\\\n\\midrule\n")
prev = None
for label, fam, d in rows:
    if prev is not None and label != prev:
        body += "\\midrule\n"
    prev = label
    body += (f"{label} & {fam} & {len(d)} & {d.n.min()}--{d.n.max()} & "
             f"{d.m.min()}--{d.m.max()} & {d.density.mean():.2f} & {int(d.is_line_graph.sum())}\\\\\n")
body += "\\bottomrule\n\\end{tabular}\n"
write("tab_sets", body)

# ------------------------------------------------------------ Table 2
body = ("\\begin{tabular}{llrrrrrrr}\n\\toprule\nSet & Family & $\\omega$ & $\\chi$ & "
        "$\\chi>\\omega$ [\\%] & $\\max(\\chi-\\omega)$ & $\\max\\chi/\\omega$ & Degen. & $D_N$\\\\\n\\midrule\n")
prev = None
for label, fam, d in rows:
    if prev is not None and label != prev:
        body += "\\midrule\n"
    prev = label
    k = d[d.chi.notna()]
    body += (f"{label} & {fam} & {d.omega.mean():.1f} & {k.chi.mean():.1f} & "
             f"{100 * (k.chi_minus_omega > 0).mean():.1f} & {int(k.chi_minus_omega.max())} & "
             f"{k.chi_over_omega.max():.2f} & {d.degeneracy.mean():.1f} & "
             f"{d.neighborhood_density.mean():.2f}\\\\\n")
body += "\\bottomrule\n\\end{tabular}\n"
write("tab_params", body)
say("== known chi ==", {l + "/" + f: (int(d.chi.notna().sum()), len(d)) for l, f, d in rows})

# ------------------------------------------------------------ Table 3
def solved(d, col):
    return 100.0 * (d[col + "_status"] == "optimal").mean()


body = ("\\begin{tabular}{llrrrrrrr}\n\\toprule\n & & \\multicolumn{4}{c}{Exact DSATUR} & "
        "\\multicolumn{3}{c}{\\textsc{V-edge} model}\\\\\n\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}\n"
        "Set & Family & Solved [\\%] & Median [ms] & Max [s] & Median nodes & "
        "Solved [\\%] & Median [ms] & Max [s]\\\\\n\\midrule\n")
prev = None
for label, fam, d in rows:
    if prev is not None and label != prev:
        body += "\\midrule\n"
    prev = label
    body += (f"{label} & {fam} & {solved(d, 'dsatur'):.1f} & {ms(d.dsatur_runtime.median(), 2)}{'$^{\\dagger}$' if solved(d, 'dsatur') <= 50 else ''} & "
             f"{d.dsatur_runtime.max():.2f} & {int(d.dsatur_search_nodes.fillna(0).median())} & "
             f"{solved(d, 'vedge'):.1f} & {ms(d.vedge_runtime.median())} & {d.vedge_runtime.max():.2f}\\\\\n")
body += "\\bottomrule\n\\end{tabular}\n"
write("tab_solvers", body)
for label, fam, d in rows:
    say(f"solved {label}/{fam}: dsatur {solved(d, 'dsatur'):.1f}% vedge {solved(d, 'vedge'):.1f}%",
        "dsatur zero-node share:", round(float((d.dsatur_search_nodes.fillna(0) == 0).mean()), 3))

# ------------------------------------------------------------ Table 4 (line graphs: L(H) vs H)
L = A[(A.vedge_status == "optimal") & (A.estar_status == "optimal") & (A.vclique_status == "optimal")].copy()
say("== set A: all three models solved on", len(L), "of", len(A))
L["S"] = L.vedge_runtime / L.estar_runtime
L["S_clique"] = (L.vclique_runtime + L.vclique_prep_time) / L.estar_runtime
L["S_total"] = L.vedge_runtime / (L.estar_runtime + L.root_reconstruction_time)
bins = [0, 1.5, 3, 6, 100]
labels = ["$R\\le 1.5$", "$1.5<R\\le 3$", "$3<R\\le 6$", "$R>6$"]
L["Rbin"] = pd.cut(L.root_ratio, bins=bins, labels=labels)


def t4row(name, d):
    w = stats.wilcoxon(np.log(d.vedge_runtime), np.log(d.estar_runtime)).pvalue if len(d) > 10 else float("nan")
    return (f"{name} & {len(d)} & {ms(d.vedge_runtime.median())} & "
            f"{ms((d.vclique_runtime + d.vclique_prep_time).median())} & {ms(d.estar_runtime.median())} & "
            f"{ms(d.root_reconstruction_time.median())} & {d.S.median():.2f} & {gmean(d.S):.2f} & "
            f"{d.S.max():.1f} & {100 * (d.S > 1).mean():.0f} & {d.S_total.median():.2f}\\\\\n"), w


body = ("\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n & & \\multicolumn{4}{c}{Median time [ms]} & "
        "\\multicolumn{4}{c}{$S=T_V/T_E$} & \\\\\n\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n"
        "Group & Graphs & $T_V$ & $T_C$ & $T_E$ & $T_{\\mathrm{rec}}$ & Median & Geo.\\ mean & Max & "
        "$S>1$ [\\%] & $S_{\\mathrm{tot}}$\\\\\n\\midrule\n")
for src in ORDER_A:
    r, w = t4row(NAMES[src], L[L.source == src])
    body += r
    say("wilcoxon", src, w)
body += "\\midrule\n"
for lab in labels:
    d = L[L.Rbin == lab]
    if len(d):
        body += t4row(lab, d)[0]
body += "\\midrule\n"
r, w = t4row("All line graphs", L)
body += r + "\\bottomrule\n\\end{tabular}\n"
write("tab_speedup", body)
say("wilcoxon all (log T_V vs log T_E): p =", w)
say("S overall: median", L.S.median(), "gmean", gmean(L.S), "share>1", (L.S > 1).mean(),
    "max", L.S.max(), "q1,q3", L.S.quantile([.25, .75]).tolist())
say("S_clique overall: median", L.S_clique.median(), "gmean", gmean(L.S_clique), "share>1", (L.S_clique > 1).mean())
say("S_total overall: median", L.S_total.median(), "gmean", gmean(L.S_total), "share>1", (L.S_total > 1).mean())
say("sum of times [s]: T_V", L.vedge_runtime.sum(), "T_C", (L.vclique_runtime + L.vclique_prep_time).sum(),
    "T_E", L.estar_runtime.sum(), "T_rec", L.root_reconstruction_time.sum(), "T_omega", L.omega_time.sum())
say("max times [s]: T_V", L.vedge_runtime.max(), "T_C", L.vclique_runtime.max(), "T_E", L.estar_runtime.max())
say("constraints median: vedge", L.vedge_constraints.median(), "vclique", L.vclique_constraints.median(),
    "estar", L.estar_constraints.median())
for x in ("root_ratio", "root_Delta", "n", "m"):
    rho, p = stats.spearmanr(L[x], L.S)
    say(f"spearman S vs {x}: rho={rho:.3f} p={p:.2e}")
say("vedge calls distribution:", L.vedge_calls.value_counts().to_dict())
say("bipartite roots:", int(L.root_bipartite.sum()), "of which class I:",
    int(((L.root_bipartite == 1) & (L.root_class == "I")).sum()))

# ------------------------------------------------------------ Table 5 (class I / II)
body = ("\\begin{tabular}{llrrrrrrr}\n\\toprule\n & & & \\multicolumn{2}{c}{Exact DSATUR} & "
        "\\multicolumn{3}{c}{Median model time [ms]} & \\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}\n"
        "Class of $H$ & Family & Graphs & Solved [\\%] & Median nodes & $T_V$ & $T_C$ & $T_E$ & Overfull [\\%]\\\\\n\\midrule\n")
for cls in ("I", "II"):
    for src in ORDER_A + ["all"]:
        d = L[L.root_class == cls] if src == "all" else L[(L.root_class == cls) & (L.source == src)]
        if len(d) == 0:
            continue
        nm = "all families" if src == "all" else NAMES[src]
        body += (f"{cls} & {nm} & {len(d)} & {solved(d, 'dsatur'):.1f} & "
                 f"{int(d.dsatur_search_nodes.median())} & {ms(d.vedge_runtime.median())} & "
                 f"{ms((d.vclique_runtime + d.vclique_prep_time).median())} & {ms(d.estar_runtime.median())} & "
                 f"{100 * d.root_overfull.mean():.0f}\\\\\n")
    if cls == "I":
        body += "\\midrule\n"
body += "\\bottomrule\n\\end{tabular}\n"
write("tab_classes", body)
say("class counts by source:", pd.crosstab(L.source, L.root_class).to_dict())
say("class II and overfull:", pd.crosstab(L.root_class, L.root_overfull).to_dict())
# matched comparison class I vs II within regular graphs of the same degree
reg = L[L.source.isin(["regular", "subdivided"])]
for col in ("vedge_runtime", "estar_runtime"):
    a, b = reg[reg.root_class == "I"][col], reg[reg.root_class == "II"][col]
    say(f"regular+subdivided {col}: median I {a.median():.4f} II {b.median():.4f} "
        f"MWU p={stats.mannwhitneyu(a, b).pvalue:.2e}")

# ------------------------------------------------------------ Table 6 (Spearman)
params = [("n", "$n$"), ("m", "$m$"), ("density", "density"), ("max_degree", "$\\Delta(G)$"),
          ("omega", "$\\omega(G)$"), ("chi_minus_omega", "$\\chi-\\omega$"),
          ("degeneracy", "degeneracy"), ("neighborhood_density", "$D_N(G)$")]
targets = [("A", L, "vedge_runtime", "$T_V$"), ("A", L, "estar_runtime", "$T_E$"),
           ("A", A, "dsatur_search_nodes", "DSATUR nodes"),
           ("B", B[B.vedge_status == "optimal"], "vedge_runtime", "$T_V$"),
           ("B", B, "dsatur_search_nodes", "DSATUR nodes")]
corr = {}
body = ("\\begin{tabular}{lrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{Set A (line graphs)} & "
        "\\multicolumn{2}{c}{Set B (non-line)}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\nParameter")
for _, _, _, tn in targets:
    body += f" & {tn}"
body += "\\\\\n\\midrule\n"
for p, pn in params:
    body += pn
    for sname, D, t, tn in targets:
        d = D[D[p].notna() & D[t].notna()]
        if d[p].nunique() < 2:
            body += " & --"
            continue
        rho, pv = stats.spearmanr(d[p], d[t])
        corr[(sname, t, p)] = rho
        star = "" if pv < 0.001 else "$^{\\circ}$"
        body += f" & ${rho:+.2f}${star}"
    body += "\\\\\n"
body += "\\bottomrule\n\\end{tabular}\n"
write("tab_corr", body)
say("== correlations ==")
for k, v in corr.items():
    say(k, round(v, 3))

# ------------------------------------------------------------ RQ1: line vs non-line, matched n
say("== RQ1 (matched n range) ==")
lo, hi = int(B.n.min()), int(B.n.max())
Am = A[(A.n >= lo) & (A.n <= hi)]
for name, D in (("A line", Am), ("B non-line", B)):
    say(name, "n in", (lo, hi), "graphs", len(D), "density", round(D.density.mean(), 3),
        "dsatur solved %", round(solved(D, "dsatur"), 1), "vedge solved %", round(solved(D, "vedge"), 1),
        "median T_V", round(D.vedge_runtime.median(), 4),
        "median dsatur nodes", D.dsatur_search_nodes.median(),
        "share gap>0", round(float((D.chi_minus_omega > 0).mean()), 3))
say("MWU T_V line vs non-line (matched n):",
    stats.mannwhitneyu(Am.vedge_runtime, B.vedge_runtime).pvalue)
say("B by source & n: dsatur solved",
    B.groupby(["source", "n"]).apply(lambda d: round(solved(d, "dsatur"), 1), include_groups=False).to_dict())
say("B by source & n: vedge solved",
    B.groupby(["source", "n"]).apply(lambda d: round(solved(d, "vedge"), 1), include_groups=False).to_dict())
say("B gap by source:", B.groupby("source").chi_reference.count().to_dict(),
    (B.chi_reference - B.omega).groupby(B.source).agg(["mean", "max"]).round(2).to_dict())
say("B vedge calls:", B.groupby("source").vedge_calls.mean().round(2).to_dict())

# ------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e3e3e0", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "font.family": "serif",
                     "pdf.fonttype": 42})
COL = {"er": "#0072B2", "bipartite": "#E69F00", "tree": "#009E73", "regular": "#CC79A7",
       "subdivided": "#D55E00", "co-bipartite": "#56B4E9", "co-triangle-free": "#000000"}
MRK = {"er": "o", "bipartite": "s", "tree": "^", "regular": "D", "subdivided": "v",
       "co-bipartite": "P", "co-triangle-free": "X"}
TL = float(max(A.dsatur_runtime.max(), B.dsatur_runtime.max()))

# Fig 2: exact colouring time vs n (line vs non-line)
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
for ax, col, title in ((axes[0], "dsatur_runtime", "Exact DSATUR"), (axes[1], "vedge_runtime", "V-edge model")):
    for D, order in ((A, ORDER_A), (B, ORDER_B)):
        for src in order:
            d = D[D.source == src]
            ax.scatter(d.n, d[col].clip(lower=1e-5), s=9, marker=MRK[src], color=COL[src],
                       alpha=0.55, linewidths=0, label=PLAIN[src])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of vertices $n$ of $G$")
    ax.set_title(title, fontsize=9)
    ax.axhline(30, color="#888888", lw=0.7, ls="--")
axes[0].set_ylabel("running time [s]")
axes[1].text(0.98, 0.93, "time limit", transform=axes[1].transAxes, ha="right", va="top", fontsize=7, color="#555555")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, markerscale=1.6, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.11, 1, 1))
fig.savefig(f"{FIG}/fig_time_vs_n.pdf")
plt.close(fig)

# Fig 3: distribution of chi - omega
groups = [("C: line", C[C.family == "line"]), ("C: non-line", C[C.family != "line"]),
          ("A: line", A), ("B: co-bipartite", B[B.source == "co-bipartite"]),
          ("B: co-triangle-free", B[B.source == "co-triangle-free"])]
fig, ax = plt.subplots(figsize=(6.2, 2.6))
gbins = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5 or more", 5, 10 ** 6)]
shades = plt.cm.Blues(np.linspace(0.2, 0.95, len(gbins)))
left = np.zeros(len(groups))
for (glab, glo, ghi), shade in zip(gbins, shades):
    vals = []
    for name, d in groups:
        gap = d.chi_minus_omega.dropna()
        vals.append(100.0 * ((gap >= glo) & (gap <= ghi)).mean())
    ax.barh(range(len(groups)), vals, left=left, color=shade, edgecolor="white", linewidth=1.2,
            label=glab, height=0.62)
    left += np.array(vals)
maxgap = len(gbins) - 1
ax.set_yticks(range(len(groups)))
ax.set_yticklabels([g[0] for g in groups])
ax.invert_yaxis()
ax.set_xlabel("share of graphs [%]")
ax.set_xlim(0, 100)
ax.grid(axis="y", visible=False)
ax.legend(title="$\\chi(G)-\\omega(G)$", ncol=maxgap + 1, fontsize=7.5, title_fontsize=8,
          loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=1.0, handlelength=1.2)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_gap.pdf")
plt.close(fig)

# Fig 4: speedup by family (box plots)
fig, ax = plt.subplots(figsize=(6.2, 2.7))
data = [L[L.source == s].S.values for s in ORDER_A]
bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                flierprops=dict(marker=".", markersize=3, alpha=0.5),
                medianprops=dict(color="black", lw=1.2))
for patch, s in zip(bp["boxes"], ORDER_A):
    patch.set_facecolor(COL[s])
    patch.set_alpha(0.75)
ax.set_xticklabels([PLAIN[s] for s in ORDER_A])
ax.set_yscale("log")
ax.axhline(1, color="#444444", lw=0.8, ls="--")
ax.set_ylabel("speed-up $S=T_V/T_E$")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_speedup_box.pdf")
plt.close(fig)

# Fig 5: speedup vs R
fig, ax = plt.subplots(figsize=(6.2, 3.0))
for s in ORDER_A:
    d = L[L.source == s]
    ax.scatter(d.root_ratio, d.S, s=11, marker=MRK[s], color=COL[s], alpha=0.6, linewidths=0, label=PLAIN[s])
ax.set_yscale("log")
ax.axhline(1, color="#444444", lw=0.8, ls="--")
ax.set_xlabel("$R=|E(H)|/|V(H)|$")
ax.set_ylabel("speed-up $S=T_V/T_E$")
ax.legend(fontsize=7.5, ncol=2, markerscale=1.6, loc="lower right")
fig.tight_layout()
fig.savefig(f"{FIG}/fig_speedup_vs_R.pdf")
plt.close(fig)

# Fig 6: correlations
fig, ax = plt.subplots(figsize=(6.2, 2.8))
series = [("A", "vedge_runtime", "$T_V$ (set A)", "#0072B2", ""),
          ("A", "estar_runtime", "$T_E$ (set A)", "#E69F00", "//"),
          ("A", "dsatur_search_nodes", "DSATUR nodes (set A)", "#009E73", ".."),
          ("B", "dsatur_search_nodes", "DSATUR nodes (set B)", "#555555", "xx")]
w = 0.2
x = np.arange(len(params))
for i, (sname, t, lab, c, hatch) in enumerate(series):
    vals = [corr.get((sname, t, p), np.nan) for p, _ in params]
    ax.bar(x + (i - 1.5) * w, vals, w * 0.92, color=c, label=lab, hatch=hatch, edgecolor="white", linewidth=0.4)
ax.set_xticks(x)
ax.set_xticklabels([pn for _, pn in params], fontsize=8)
ax.axhline(0, color="black", lw=0.6)
ax.set_ylabel("Spearman's $\\rho$")
ax.set_ylim(-1, 1.18)
ax.grid(axis="x", visible=False)
ax.legend(fontsize=7.5, ncol=4, loc="upper center", columnspacing=1.0, handlelength=1.4)
fig.tight_layout()
fig.savefig(f"{FIG}/fig_corr.pdf")
plt.close(fig)
log.close()
