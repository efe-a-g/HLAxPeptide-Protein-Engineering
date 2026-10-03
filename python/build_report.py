#!/usr/bin/env python3
"""
Build the self-contained local HTML report in outputs/report/.

Reads outputs/ablation/results.jsonl, the per-allele CSVs, the noise-floor JSON
and outputs/report/structures.json, and emits index.html with everything
embedded. Charts are hand-built SVG drawn by vanilla JS; the 3D viewers pull
coordinates from the RCSB PDB at view time. Nothing is published anywhere.

Summary/delta logic is imported from aggregate.py rather than duplicated, so
the page and the console tables can never disagree -- including the rule that
training-protocol variants (25 vs 100 epochs, 60 vs 256 units, PCA) are never
pooled.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from aggregate import (BASE_VARIANT, METRICS, SPLIT_ORDER, load, paired_deltas,
                       pretty, summarize)

BASELINE = "BLOSUM-pep | BLOSUM-HLA"

SPLIT_META = {
    "random": {
        "name": "Random split",
        "blurb": "Rows shuffled. Every test allele, and ~49% of test peptides, "
                 "also appear in training. The easiest split and the least "
                 "informative. Baseline seed SD 0.0013 — very stable.",
    },
    "cluster": {
        "name": "Peptide-grouped split",
        "blurb": "No peptide appears in both train and test (audited: 0 shared, "
                 "every seed). Tests generalisation to unseen peptides; all "
                 "alleles are still seen. Baseline seed SD 0.0133.",
    },
    "allele": {
        "name": "Leave-allele-out split",
        "blurb": "15 held-out alleles, zero allele overlap (audited). The real "
                 "test of the premise — but baseline seed SD is 0.0812, about "
                 "61x the random split, because which alleles are held out "
                 "matters more than the model. Read differences here with care.",
    },
}

VARIANT_META = {
    BASE_VARIANT: {
        "name": "Published protocol",
        "blurb": "25 epochs, 60 hidden units — identical to train_baseline.py.",
    },
    "ep100": {
        "name": "100 epochs",
        "blurb": "Four times the training budget. Tests whether a "
                 "high-dimensional ESM-2 input was simply undertrained.",
    },
    "h256": {
        "name": "256 hidden units",
        "blurb": "A wider head. Tests whether 60 units bottlenecked the "
                 "larger inputs.",
    },
    "pca180": {
        "name": "PCA to 180 dims",
        "blurb": "Learned blocks projected to BLOSUM-peptide width, so input "
                 "widths match and representation quality is separated from "
                 "first-layer parameter count.",
    },
}


def group_of(pep, hla):
    """Family of an arm -> drives colour (3 validated palette slots)."""
    if pep == "none" or hla == "none":
        return "ablation"
    if pep == "blosum_pep" and hla == "blosum_hla":
        return "baseline"
    if hla == "onehot_hla":
        return "control"
    learned_pep = pep.startswith("esm")
    learned_hla = any(p.startswith(("boltz", "esm")) for p in hla.split("+"))
    if learned_pep or learned_hla:
        return "foundation"
    return "control"


def minify_pdb(s, src_dir="outputs/report/structures"):
    """
    Read a PDB file and reduce it to just what the viewer draws, so the
    coordinates can be EMBEDDED in the page.

    This has to be embedded rather than fetched: Chrome refuses a file:// page
    permission to read sibling file:// resources, so `loadFile("structures/X.pdb")`
    fails silently with a blank panel when the report is opened by double-click.
    Embedding removes the fetch entirely.

    Kept: ATOM records for the three chains actually rendered, plus the TER/END
    framing. Dropped: every other chain (TCR chains and the second copy in the
    asymmetric unit), ANISOU (which roughly doubles the size of a high-resolution
    file and is never drawn), waters, and all header/remark metadata.
    """
    path = os.path.join(src_dir, f"{s['pdb_id']}.pdb")
    if not os.path.isfile(path):
        return None
    keep_chains = {c for c in (s.get("chain_hla"), s.get("chain_b2m"),
                               s.get("chain_peptide")) if c}
    out, seen_model = [], False
    with open(path, "r", errors="replace") as f:
        for line in f:
            rec = line[:6]
            if rec == "ENDMDL":
                break               # first model only
            if rec == "MODEL ":
                if seen_model:
                    break
                seen_model = True
                continue
            if rec not in ("ATOM  ", "HETATM"):
                continue
            if line[17:20].strip() in ("HOH", "DOD"):
                continue
            if line[21] not in keep_chains:
                continue
            # Keep only the primary altloc so the viewer does not draw doubles.
            if line[16] not in (" ", "A"):
                continue
            out.append(line.rstrip("\n").rstrip())
    out.append("END")
    return "\n".join(out)


def load_per_allele(pep, hla, split, variant):
    """Average the per-allele CSVs for one arm across seeds."""
    if variant != BASE_VARIANT:
        return None
    frames = []
    for p in sorted(glob.glob(
            f"outputs/ablation/per_allele/{pep}__{hla}__{split}__s*.csv")):
        base = os.path.basename(p)
        # Base-variant files are '<...>__s<seed>.csv'; tagged runs carry a
        # further '__<tag>' and must not be mixed in.
        stem = base[:-4]
        if not stem.split("__")[-1].startswith("s"):
            continue
        t = pd.read_csv(p)
        if len(t):
            frames.append(t)
    if not frames:
        return None
    allc = pd.concat(frames)
    return allc.groupby("allele").agg(
        n_samples=("n_samples", "mean"), pcc=("pcc", "mean"),
        pcc_sd=("pcc", "std"), n_seeds=("pcc", "size")).reset_index()


def build_payload(results_path, noise_path, struct_path):
    df = load(results_path)
    df = df[df["variant"] != "noise_init"]

    # --pca_dim projects only the LEARNED blocks, so a pure-BLOSUM arm under
    # pca180 would be bit-identical to the same arm at base. Rather than spend
    # runs re-computing it, the base baseline is carried into the pca variant so
    # the dimension-matched arm has its correct comparator. This is an identity,
    # not an approximation -- but it is only valid for pca-style variants, never
    # for ones that change the training protocol itself (epochs, width).
    for v in sorted(set(df["variant"])):
        if not v.startswith("pca") or BASELINE in set(df[df["variant"] == v]["config"]):
            continue
        carried = df[(df["variant"] == BASE_VARIANT)
                     & (df["config"] == BASELINE)].copy()
        if carried.empty:
            continue
        carried["variant"] = v
        df = pd.concat([df, carried], ignore_index=True)
    # Drop any protocol variant that was abandoned part-way. A variant needs at
    # least two arms with >=3 seeds each to support a comparison; one arm alone
    # is just an isolated number with nothing to read it against. (The 256-unit
    # probe was stopped once the 100-epoch probe had answered the same question,
    # leaving only its baseline, so it is dropped here.)
    keep = []
    for v, g in df.groupby("variant"):
        complete = (g.groupby("config")["seed"].nunique() >= 3).sum()
        if complete >= 2:
            keep.append(v)
    dropped = sorted(set(df["variant"]) - set(keep))
    if dropped:
        print(f"[*] dropping incomplete variants from the report: {dropped}")
    df = df[df["variant"].isin(keep)]
    df["group"] = [group_of(p, h) for p, h in zip(df["pep"], df["hla"])]

    summ = summarize(df)
    gmap = {(p, h): group_of(p, h) for p, h in zip(df["pep"], df["hla"])}
    cfgmap = {pretty(p, h): (p, h) for p, h in zip(df["pep"], df["hla"])}
    summ["group"] = [gmap[cfgmap[c]] for c in summ["config"]]
    dd = paired_deltas(df, BASELINE)
    if not dd.empty:
        dd["group"] = [gmap[cfgmap[c]] for c in dd["config"]]

    # ---- per-allele: baseline vs the best foundation arm (base variant) ----
    per_allele = {}
    base_summ = summ[summ["variant"] == BASE_VARIANT]
    for split in SPLIT_ORDER:
        s = base_summ[base_summ["split"] == split]
        if s.empty:
            continue
        fm = s[s["group"] == "foundation"]
        best_cfg = (fm.loc[fm["mean_allele_pcc"].idxmax(), "config"]
                    if not fm.empty and fm["mean_allele_pcc"].notna().any() else None)
        bp, bh = cfgmap[BASELINE]
        b = load_per_allele(bp, bh, split, BASE_VARIANT)
        if b is None:
            continue
        merged = b.rename(columns={"pcc": "pcc_base", "pcc_sd": "pcc_base_sd"})
        if best_cfg:
            fp, fh = cfgmap[best_cfg]
            f = load_per_allele(fp, fh, split, BASE_VARIANT)
            if f is not None:
                merged = merged.merge(
                    f[["allele", "pcc", "pcc_sd"]].rename(
                        columns={"pcc": "pcc_fm", "pcc_sd": "pcc_fm_sd"}),
                    on="allele", how="outer")
        per_allele[split] = {
            "baseline": BASELINE, "foundation": best_cfg,
            "rows": json.loads(merged.to_json(orient="records")),
        }

    structures = json.load(open(struct_path)) if os.path.isfile(struct_path) else []
    for s in structures:
        s["pdb_text"] = minify_pdb(s)
    noise = json.load(open(noise_path)) if os.path.isfile(noise_path) else {}

    variants = [BASE_VARIANT] + sorted(
        v for v in summ["variant"].unique() if v != BASE_VARIANT)

    return {
        "summary": json.loads(summ.to_json(orient="records")),
        "deltas": json.loads(dd.to_json(orient="records")) if not dd.empty else [],
        "per_allele": per_allele,
        "n_runs": int(len(df)),
        "structures": structures, "noise": noise,
        "baseline": BASELINE, "base_variant": BASE_VARIANT,
        "variants": variants, "variant_meta": VARIANT_META,
        "split_meta": SPLIT_META, "split_order": SPLIT_ORDER,
        "metrics": METRICS,
        "seeds": sorted(int(s) for s in df["seed"].unique()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/ablation/results.jsonl")
    ap.add_argument("--noise", default="outputs/ablation/noise_floor.json")
    ap.add_argument("--structures", default="outputs/report/structures.json")
    ap.add_argument("--out", default="outputs/report/index.html")
    ap.add_argument("--template", default="python/report_template.html")
    args = ap.parse_args()

    payload = build_payload(args.results, args.noise, args.structures)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def clean(o):
        """NaN/Inf -> None so the blob is valid JSON for JSON.parse."""
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return clean(float(o))
        return o

    tpl = open(args.template, encoding="utf-8").read()
    blob = json.dumps(clean(payload), separators=(",", ":"), allow_nan=False,
                      default=lambda o: None)
    # The blob sits inside <script type="application/json">; neutralise any
    # sequence that could close that element early.
    blob = blob.replace("</", "<\\/")
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(tpl.replace("/*__DATA__*/", blob))

    kb = os.path.getsize(args.out) / 1024
    print(f"[+] wrote {args.out} ({kb:.0f} KB) | {payload['n_runs']} runs, "
          f"{len(payload['summary'])} summary rows, {len(payload['deltas'])} deltas, "
          f"variants {payload['variants']}, {len(payload['structures'])} structures")


if __name__ == "__main__":
    main()
