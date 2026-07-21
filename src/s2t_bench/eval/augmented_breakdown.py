# variant_breakdown.py
import collections
import json
import statistics

data = json.load(open("results/whisper_aug/whisper_eval.json"))
for m in data:
    by = collections.defaultdict(list)
    for s in m["per_sample"]:
        by[s["variant"]].append(s["wer"])
    print(f"\n=== {m['model']} ===")
    for variant, wers in sorted(by.items(), key=lambda kv: statistics.mean(kv[1])):
        print(f"  {variant:16s} WER {statistics.mean(wers):.3f}  (n={len(wers)})")