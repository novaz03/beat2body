"""Per-genre comparison of the evaluated models, from the eval_test_<tag>.json files on Hugging Face (light: no GPU).

    python scripts/summarize_genres.py                     # models of the per-genre queue, tag feet2_raw_genre
    python scripts/summarize_genres.py --out reports/2026-11-24_genres/README.md

Needs read access to SizheZ03/beat2body-runs (`hf auth login` or HF_TOKEN). Writes markdown:
  1. genre difficulty: stance drift and BoneCV relative to the genre's real values (comparable across genres), with
     the FIDs next to the genre's real reference (real training dances of the same genre, same number of windows;
     FIDs are not comparable across genres);
  2. one table per genre: every model next to the real reference values.
"""
import argparse
import json
import os
import statistics

from huggingface_hub import HfApi, hf_hub_download

HF_REPO = "SizheZ03/beat2body-runs"
COLS = [  # (key, real reference key, header, format)
    ("FID_k", "real_train_FID_k", "FID_k", "{:.1f}"),
    ("FID_p", "real_train_FID_p", "FID_p", "{:.1f}"),
    ("Div_k", "real_Div_k", "Div_k", "{:.2f}"),
    ("speed", None, "speed", "{:.2f}"),
    ("BeatPow", None, "BeatPow", "{:.2f}"),
    ("BoneCV", "real_BoneCV", "BoneCV", "{:.3f}"),
    ("stance_drift", "real_stance_drift", "stance drift", "{:.3f}"),
    ("swing_speed", "real_swing_speed", "swing", "{:.3f}"),
    ("no_foot_planted", "real_no_foot_planted", "no foot", "{:.0%}"),
    ("root_jitter", "real_root_jitter", "jitter", "{:.4f}"),
    ("flight_acc_y_median", "real_flight_acc_y_median", "fall rate", "{:.4f}"),
    ("flights", "real_flights", "jumps", "{}"),
    ("penetration_mean", "real_penetration_mean", "below floor", "{:.4f}"),
    ("drift_height", "real_drift_height", "height drift", "{:.3f}"),
]


def fmt(f, v):
    if v is None:
        return "–"
    try:
        return f.format(v)
    except (TypeError, ValueError):
        return "–"


def load(tag, token=None):
    api = HfApi(token=token)
    queue = json.load(open(hf_hub_download(HF_REPO, "evals/queue.json", token=token, force_download=True)))
    files = set(api.list_repo_files(HF_REPO))
    out = {}
    for e in queue:
        if e["tag"] != tag:
            continue
        path = f"runs/{e['job']}/eval_test_{tag}.json"
        if path not in files:
            print(f"not evaluated yet: {e['name']} ({path})")
            continue
        d = json.load(open(hf_hub_download(HF_REPO, path, token=token)))
        if d.get("by_genre"):
            out[e["name"]] = d["by_genre"]
    return out


def report(models):
    genres = sorted({g for m in models.values() for g in m})
    lines = ["# Per-genre comparison", "",
             f"Models: {', '.join(models)}. Real reference = real training dances of the same genre, as many windows "
             "as the genre's test windows. Speed and BeatPow are relative to the genre's real test dances (1 = real).",
             "", "## Genre difficulty", "",
             "Ranked by how far the models' feet and bodies are from real dance in that genre: stance drift (a planted "
             "foot sliding) and BoneCV (body size changing), each divided by the genre's real value and averaged over "
             "the models (1 = like real). These ratios are comparable across genres. FIDs are shown with the genre's "
             "real reference, but FID differences between genres are not comparable: the reference ranges from about 2 "
             "to 33 (how much the test dances differ from the training dances of the genre) and windows per genre "
             "vary.", "",
             "| genre | windows | stance drift ÷ real | BoneCV ÷ real | FID_k: real ref / models | "
             "FID_p: real ref / models |",
             "|---|---:|---:|---:|---|---|"]
    rows = []
    for g in genres:
        vs = [m[g] for m in models.values() if g in m]
        sd = [v["stance_drift"] / v["real_stance_drift"] for v in vs if v.get("real_stance_drift")]
        bc = [v["BoneCV"] / v["real_BoneCV"] for v in vs if v.get("real_BoneCV")]
        fk = ", ".join(f"{v['FID_k']:.0f}" for v in vs)
        fp = ", ".join(f"{v['FID_p']:.0f}" for v in vs)
        rows.append((statistics.mean(sd) if sd else float("nan"), g, vs[0]["windows"],
                     statistics.mean(bc) if bc else float("nan"),
                     f"{fmt('{:.1f}', vs[0].get('real_train_FID_k'))} / {fk}",
                     f"{fmt('{:.1f}', vs[0].get('real_train_FID_p'))} / {fp}"))
    for sd, g, n, bc, fk, fp in sorted(rows, reverse=True):
        lines.append(f"| {g} | {n} | {sd:.1f} | {bc:.2f} | {fk} | {fp} |")
    lines += ["", f"FID columns list the models in this order: {', '.join(models)}."]
    for g in genres:
        lines += ["", f"## {g}", "", "| | " + " | ".join(h for _, _, h, _ in COLS) + " |",
                  "|---|" + "---:|" * len(COLS)]
        ref = next((m[g] for m in models.values() if g in m), {})
        lines.append("| **real** | " + " | ".join(fmt(f, ref.get(r)) if r else "1" for _, r, _, f in COLS) + " |")
        for name, m in models.items():
            v = m.get(g, {})
            lines.append(f"| {name} | " + " | ".join(fmt(f, v.get(k)) for k, _, _, f in COLS) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="feet2_raw_genre")
    ap.add_argument("--out", default=None, help="markdown file (default: print)")
    a = ap.parse_args(argv)
    models = load(a.tag, os.environ.get("HF_TOKEN"))
    if not models:
        raise SystemExit("no per-genre results yet")
    text = report(models)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(text)
        print("wrote", a.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
