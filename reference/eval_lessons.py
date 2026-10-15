"""Full held-out evaluation on the lesson files.  python3 eval_lessons.py <lesson_dir> <ckpt> [out.json]"""
import glob, json, sys
from nova2_ref import Nova2
import lessons as L
from train_lessons import evaluate
items = [it for f in sorted(glob.glob(sys.argv[1] + "/*.txt")) for it in L.parse_file(f)]
test = [it for it in items if it["test"]]
m, tok = Nova2.load(sys.argv[2])
r = evaluate(m, tok, test)
r["step"] = m.step; r["held_out_units"] = len(test)
print(json.dumps(r))
if len(sys.argv) > 3: json.dump(r, open(sys.argv[3], "w"), indent=1)
