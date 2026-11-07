"""Audit bundled data and upstream reward without importing the training stack."""
import contextlib, importlib.util, io, json, os
from pathlib import Path
import pyarrow.parquet as pq
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rlla", root/"upstream/verl/utils/reward_score/rlla.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
for key in ("WITHLENGTH", "REFINEDREWARD", "COARSEREWARD", "CORRECTMAX1", "MAX1STEP30MAX3", "SCHEDULEREWARD", "SCHEDULELENGTH", "INTERMEDIATEREWARD"):
    os.environ[key] = "0"
os.environ["EXPERIMENT_NAME"] = "toolrl-qwen-verification"
report = {"data": {}, "reward_cases": {}}
for split in ("train", "test"):
    table = pq.read_table(root/f"upstream/dataset/rlla_4k/{split}.parquet")
    assert {"prompt", "reward_model", "data_source"} <= set(table.column_names)
    rows = table.to_pylist()
    with contextlib.redirect_stdout(io.StringIO()):
        scores = [r.compute_score(x["reward_model"]["ground_truth"], x["reward_model"]["ground_truth"])[0] for x in rows]
    report["data"][split] = {"rows": len(rows), "columns": table.column_names, "self_match_min": min(scores), "self_match_max": max(scores)}
gt = '<think>reason</think>\n<tool_call>\n{"name":"weather","parameters":{"city":"Beijing"}}\n</tool_call>'
examples = {"exact": gt, "wrong_value": gt.replace('Beijing','Shanghai'), "wrong_tool": gt.replace('weather','search'), "malformed": 'bad output'}
with contextlib.redirect_stdout(io.StringIO()):
    report["reward_cases"] = {k: r.compute_score(v, gt) for k,v in examples.items()}
assert report["reward_cases"]["exact"][0] > report["reward_cases"]["wrong_value"][0] > report["reward_cases"]["wrong_tool"][0] > report["reward_cases"]["malformed"][0]
(root/"results/verification.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
