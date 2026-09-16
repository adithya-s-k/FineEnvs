"""Rebuild the two-run report, figure and uniform Trackio project from audited sources."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlencode

REPO = Path(os.environ.get("TRL_PROD", Path.cwd()))
TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import trackio_multi4 as native

PROJECT = "qwen35-2b-harbor-vs-opencode-20260916"
SPACE = "HuggingEnvs/data-agent-training-comparison-trackio"
BUCKET = "HuggingEnvs/data-agent-training-comparison-trackio"
MAIN = REPO / "experiments/async_grpo_harbor_data_agent/logs/multi4-long-prod-cont-20260915"
COMPARISON = REPO / "experiments/daytona_harness_comparison/logs"
OUTPUTS = COMPARISON / "hf-20260915/local-opencode-smoke-v4/repro/outputs"
DEFAULT_OUT = REPO / "HuggingEnvs/04-data-agent/reports/async-comparison-20260916"
ARMS = ("Harbor multi-harness", "Native OpenCode")
HARNESSES = ("opencode", "claude-code", "codex", "mini-swe-agent")
LEVELS = ("easy", "medium", "hard")
COUNTS = dict(zip(LEVELS, (33, 118, 99)))
COLORS = dict(zip(ARMS, ("#4f46e5", "#ea7c24")))


def read(path):
    return json.loads(Path(path).read_text())


def validated_score(score):
    if not all(score.get(k) is True for k in ("complete", "comparison_ready", "tito_pass", "harness_versions_match_baseline")):
        raise ValueError("Score has not passed all comparison gates")
    if score.get("graded_cells") != 1000 or set(score["harnesses"]) != set(HARNESSES):
        raise ValueError("Expected the complete four-harness cohort")
    for h in HARNESSES:
        s = score["harnesses"][h]
        if s["graded"] != 250:
            raise ValueError("Incomplete harness coverage")
        if set(s["difficulty"]) != set(LEVELS):
            raise ValueError("Missing difficulty category")
        for d, n in COUNTS.items():
            c = s["difficulty"][d]
            if c["graded"] != n or not 0 <= c["correct"] <= n:
                raise ValueError("Unexpected difficulty cohort")
        if not math.isclose(sum(c["correct"] for c in s["difficulty"].values()) / 250, s["pass_at_1"]):
            raise ValueError("Difficulty totals do not reconcile")
    if not math.isclose(sum(s["pass_at_1"] for s in score["harnesses"].values()) / 4, score["average_pass_at_1"]):
        raise ValueError("Harness average does not reconcile")
    return score


def collect():
    result = {"updated_utc": datetime.now(timezone.utc).isoformat(), "project": PROJECT,
              "space": SPACE, "runs": {}, "pending": []}
    protocol = read(REPO / "HuggingEnvs/04-data-agent/eval/baseline_protocol.json")
    baseline_file = Path(protocol["baseline_run"]) / f"job-{protocol['baseline_job']}/canonical_results.json"
    base = read(baseline_file)
    assert base["coverage_complete"]
    harbor_base = {"complete": True, "comparison_ready": True, "tito_pass": True,
        "harness_versions_match_baseline": True, "graded_cells": 1000,
        "average_pass_at_1": protocol["average_pass_at_1"], "harnesses": {}}
    assert protocol["tito"]["passed"] == protocol["tito"]["audited"] == 1000
    for h in HARNESSES:
        harbor_base["harnesses"][h] = {"graded": 250, "pass_at_1": protocol["scores"][h]["pass_at_1"],
                                     "difficulty": base["harnesses"][h]["difficulty"]}
    segments = native.training_lineage(MAIN, "80608")
    training = [row for segment in segments for row in segment["rows"]]
    sources = [{k: segment[k] for k in ("job", "start_step", "end_step", "training")} for segment in segments]
    result["runs"][ARMS[0]] = {"training": training, "lineage": sources, "evaluations": {},
        "baseline_context": "Original four-harness baseline on E2B; resumed training has documented recipe changes."}
    result["runs"][ARMS[1]] = {"training": [r for r in native.read_metrics(OUTPUTS / "local-train-opencode-80626/audit/metrics.jsonl") if "grad_norm" in r],
        "lineage": [{"job": "80626", "start_step": 1, "end_step": 1000}],
        "evaluations": {}, "baseline_context": "Four-harness Harbor/Daytona baseline; standalone native 8.4% baseline is a different evaluation and is excluded."}
    command = read(OUTPUTS / "local-train-opencode-80626/training_recipe.json")["command"]
    recipe = {}
    for i, key in enumerate(command):
        if key in {"--learning-rate", "--num-generations", "--max-inflight", "--max-staleness", "--grad-accum",
                   "--max-outstanding-rollouts", "--max-row-tokens", "--per-device-batch-size", "--agent-step-limit",
                   "--agent-timeout", "--token-budget", "--max-completion-length", "--dtype", "--top-p", "--temperature", "--save-steps"}:
            recipe[key[2:].replace("-", "_")] = command[i+1]
    recipe["atomic_rollouts"] = "--atomic-rollouts" in command
    result["runs"][ARMS[1]]["lineage"][0]["training"] = recipe
    def add(arm, step, score, path):
        validated_score(score)
        value = {"step": step, "score": score, "source": str(path),
                 "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        prior = result["runs"][arm]["evaluations"].get(step)
        if prior and prior["score"] != score:
            raise ValueError("Conflicting checkpoint evaluations")
        result["runs"][arm]["evaluations"][step] = value
    add(ARMS[0], 0, harbor_base, baseline_file)
    native_base = COMPARISON / "20260915/blackbox/canonical_scores.json"
    add(ARMS[1], 0, read(native_base), native_base)
    for root in native.evaluation_roots(MAIN):
        for directory in sorted((root / "checkpoint-evals").glob("step-*")):
            path = directory / "scores.json"
            if not path.exists() or not read(path).get("comparison_ready"):
                continue
            if read(directory / "eval_plan.json")["protocol"] != protocol:
                raise ValueError("Harbor evaluation protocol changed")
            add(ARMS[0], int(directory.name.split("-")[-1]), read(path), path)
    for evidence_path in sorted(OUTPUTS.glob("local-eval-opencode-*/checkpoint_evaluation.json")):
        evidence = read(evidence_path)
        if evidence["step"] < 50:
            continue
        path = evidence_path.parent / "canonical_scores.json"
        if path.exists() and read(path).get("comparison_ready"):
            if "/local-train-opencode-80626/" not in evidence["checkpoint_prefix"]:
                raise ValueError("Native evaluation belongs to another trainer")
            add(ARMS[1], evidence["step"], read(path), path)
    registry = DEFAULT_OUT / 'additional_harbor_runs.json'
    for arm, location in (read(registry).items() if registry.exists() else []):
        if arm in result['runs']:
            raise ValueError('Additional run would overwrite a reference run')
        root = Path(location)
        cfg = read(root/'run_config.json')
        if (cfg['model'] != protocol['model'] or cfg['model_revision'] != protocol['model_revision']
                or read(Path(cfg['evaluation']['protocol_file'])) != protocol):
            raise ValueError('Additional Harbor run has a different model or evaluation protocol')
        receipt = read(root/'submission.json') if (root/'submission.json').exists() else {}
        segments = native.training_lineage(root, receipt['training']) if receipt.get('training') else []
        result['runs'][arm] = {'training': [r for s in segments for r in s['rows']],
            'lineage': [{k:s[k] for k in ('job','start_step','end_step','training')} for s in segments],
            'evaluations': {}, 'target_steps': cfg['training']['max_steps'],
            'baseline_context': 'Fresh pinned base. Same original Harbor/E2B four-harness baseline; trains OpenCode only.'}
        add(arm, 0, harbor_base, baseline_file)
        for path in sorted((root/'checkpoint-evals').glob('step-*/scores.json')):
            if not read(path).get('comparison_ready'):
                continue
            if read(path.parent/'eval_plan.json')['protocol'] != protocol:
                raise ValueError('Additional checkpoint protocol differs from baseline')
            add(arm, int(path.parent.name.split('-')[-1]), read(path), path)
    for arm, run in result["runs"].items():
        steps = [r["step"] for r in run["training"]]
        if steps != list(range(1, max(steps, default=0) + 1)):
            raise ValueError("Training history contains gaps or duplicates")
        for step in range(100, run.get('target_steps',max(steps,default=0)) + 1, 100):
            if step not in run["evaluations"]:
                result["pending"].append({"run": arm, "step": step})
    return result


def score_metrics(score, baseline):
    metrics = {"eval/pass_at_1": score["average_pass_at_1"],
               "eval/delta_from_baseline": score["average_pass_at_1"] - baseline["average_pass_at_1"],
               "eval/graded_cells": score["graded_cells"]}
    for h in HARNESSES:
        s = score["harnesses"][h]
        metrics[f"eval/harness/{h}/pass_at_1"] = s["pass_at_1"]
        for d in LEVELS:
            c = s["difficulty"][d]
            metrics[f"eval/harness_difficulty/{h}/{d}/pass_at_1"] = c["correct"] / c["graded"]
    for d in LEVELS:
        cs = [score["harnesses"][h]["difficulty"][d] for h in HARNESSES]
        metrics[f"eval/difficulty/{d}/pass_at_1"] = sum(c["correct"] for c in cs) / sum(c["graded"] for c in cs)
    return metrics


def events(snapshot):
    records = []
    for arm, run in snapshot["runs"].items():
        config = {"model": "Qwen/Qwen3.5-2B", "training_arm": arm, "metric_axis": "optimizer step",
                  "test_tasks": 250, "eval_harnesses": list(HARNESSES), "pass_k": 1,
                  "difficulty_tasks": COUNTS, "baseline_context": run["baseline_context"],
                  "training_lineage": run["lineage"], "training_configuration_changes_preserved": True,
                  "plot_units": "scores and rewards are fractions in [0,1]; time metrics are seconds"}
        rewards, grads = [], []
        for row in run["training"]:
            metrics = native.scalars({k: v for k, v in row.items() if k != "step"}, "train/")
            reward = row.get("reward", row.get("rewards/harness_reward"))
            if reward is not None:
                rewards.append(reward)
                metrics["train/reward"] = reward
                metrics["train/reward_rolling20"] = sum(rewards[-20:]) / len(rewards[-20:])
            grads.append(int(row["grad_norm"] != 0))
            metrics["train/nonzero_gradient_rolling20"] = sum(grads[-20:]) / len(grads[-20:])
            records.append(native.event(PROJECT, arm, row["step"], metrics, config, identity="training"))
        baseline = run["evaluations"][0]["score"]
        for step, value in sorted(run["evaluations"].items()):
            records.append(native.event(PROJECT, arm, step, score_metrics(value["score"], baseline), config, identity="evaluation"))
    return records


def report(snapshot, out):
    def pct(x): return f"{100*x:.1f}%"
    arms = list(snapshot['runs'])
    counts = '; '.join(f"{a}: {len(r['training'])} steps" for a,r in snapshot['runs'].items())
    lines = ["# Harbor and OpenCode — consolidated training and pass@1", "", f"Updated: {snapshot['updated_utc']}", "",
        f"[Live Trackio dashboard](https://huggingface.co/spaces/{SPACE}) · [Overview image](comparison.png) · [Snapshot](snapshot.json)", "",
        "Qwen3.5-2B; 1,000 optimizer-step target per run. Recorded training: " + counts + ". Every accepted checkpoint has 250 fixed tasks × four harnesses = 1,000 grades. Task difficulty: 33 easy, 118 medium, 99 hard (13.2% / 47.2% / 39.6%). Scores retain first graded attempts; incomplete and failed-audit evaluations are excluded. Missing scores are not estimated.", "",
        "Baselines are separate measured cohorts: Harbor/E2B 14.6%; Harbor/Daytona 15.9% for the native OpenCode checkpoint evaluator. The standalone native OpenCode 8.4% baseline uses a different harness protocol and is excluded here. Infrastructure and training recipe histories differ; this is an observational comparison, not a controlled causal experiment.", "",
        "## Overall checkpoint curve", "", "| Checkpoint | " + ' | '.join(arms) + ' |', "| --- | " + ' | '.join('---:' for _ in arms) + ' |']
    steps = sorted({0, *range(100,1001,100), *(s for r in snapshot["runs"].values() for s in r["evaluations"])})
    for step in steps:
        cells = [pct(snapshot["runs"][a]["evaluations"][step]["score"]["average_pass_at_1"]) if step in snapshot["runs"][a]["evaluations"] else ("Pending" if step % 100 == 0 else "Not scheduled") for a in arms]
        label = "0 (baseline)" if step == 0 else str(step) + (" (recovery)" if step % 100 else "")
        lines.append(f"| {label} | {' | '.join(cells)} |")
    for arm, run in snapshot["runs"].items():
        lines += ["", "## " + arm, "", "### Overall and difficulty", "", "| Checkpoint | Overall | Easy (132 cells) | Medium (472) | Hard (396) |", "| --- | ---: | ---: | ---: | ---: |"]
        for step, value in sorted(run["evaluations"].items()):
            m = score_metrics(value["score"],run["evaluations"][0]["score"])
            lines.append(f"| {step} | " + " | ".join(pct(m[k]) for k in ["eval/pass_at_1",*[f"eval/difficulty/{d}/pass_at_1" for d in LEVELS]]) + " |")
        lines += ["", "### Harness × difficulty at every checkpoint", "", "| Checkpoint | Harness | Overall (250) | Easy (33) | Medium (118) | Hard (99) |", "| --- | --- | ---: | ---: | ---: | ---: |"]
        for step, value in sorted(run["evaluations"].items()):
            for h in HARNESSES:
                s=value["score"]["harnesses"][h]
                cells = [pct(s["pass_at_1"])] + [f"{pct(s['difficulty'][d]['correct']/COUNTS[d])} ({int(s['difficulty'][d]['correct'])}/{COUNTS[d]})" for d in LEVELS]
                lines.append(f"| {step} | {h} | " + " | ".join(cells) + " |")
        lines += ["", "### Training history", "", "| Allocation | First optimizer step | Last optimizer step |", "| --- | ---: | ---: |"]
        lines += [f"| {s['job']} | {s['start_step']} | {s['end_step']} |" for s in run["lineage"]]
        lines += ["", "### Score provenance", ""]
        lines += [f"- Step {s}: `{v['source']}`; SHA256 `{v['source_sha256']}`." for s,v in sorted(run["evaluations"].items())]
    lines += ["", "## Dashboard metric guide", "", "Both runs use identical metric names and optimizer-step axes. `eval/pass_at_1` is the overall score; `eval/difficulty/*` aggregates each difficulty; `eval/harness/*` compares each harness; `eval/harness_difficulty/*` contains all twelve intersections. `train/*` preserves recorded loss, reward, learning rate, gradient norm, entropy, KL, staleness, throughput, token, batching and rollout metrics where observed. Missing metrics are not filled with zeros. `train/reward_rolling20` and `train/nonzero_gradient_rolling20` are explicitly derived trailing windows. Raw metrics remain available. Use zero dashboard smoothing for exact checkpoint values.", "", "The independent CPU publisher refreshes every 60 seconds and admits new evaluations only after their full comparison gates pass. It never changes trainer state. Local SQLite backup, event ledger and remote exact-content verification receipts are kept alongside this report.", "",
        "Storage and deployment follow the [Trackio guide](https://huggingface.co/docs/trackio/quickstart) and [environment configuration](https://huggingface.co/docs/trackio/environment_variables).", ""]
    ids = ','.join(native.digest([PROJECT, a])[:32] for a in arms)
    for label, pattern in [('Overview', '^(eval/pass_at_1|train/reward_rolling20)$'),
                           ('Difficulty', '^eval/difficulty/'), ('Harness', '^eval/harness/'),
                           ('Harness × difficulty', '^eval/harness_difficulty/'),
                           ('Optimizer diagnostics', '^train/(loss|grad_norm|entropy|kl|learning_rate|nonzero_gradient_rolling20)$'),
                           ('Throughput and rollout diagnostics', '^train/(perf|rollout|sample|batch)/'),
                           ('All metrics', '')]:
        url = 'https://huggingenvs-data-agent-training-comparison-trackio.hf.space/?' + urlencode(
            dict(project=PROJECT, run_ids=ids, smoothing=0, metric_filter=pattern))
        lines.append(f'- [{label}]({url})')
    lines += ['', '[Download checkpoint scores as CSV](checkpoint_scores.csv)', '']
    (out / "REPORT.md").write_text("\n".join(lines))
    with (out / "checkpoint_scores.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "checkpoint", "harness", "difficulty", "correct", "graded", "pass_at_1"])
        for arm, run in snapshot["runs"].items():
            for step, value in sorted(run["evaluations"].items()):
                for h in HARNESSES:
                    for d in LEVELS:
                        c = value["score"]["harnesses"][h]["difficulty"][d]
                        writer.writerow([arm, step, h, d, int(c["correct"]), c["graded"], c["correct"]/c["graded"]])


def plot(snapshot, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MultipleLocator
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.titlesize":12,
                        "axes.titleweight":"bold", "axes.edgecolor":"#d5dce6", "text.color":"#18243a",
                        "axes.labelcolor":"#596579", "xtick.color":"#596579", "ytick.color":"#596579"})
    fig=plt.figure(figsize=(18,12),facecolor="#f5f7fb")
    gs=fig.add_gridspec(3,12,left=.055,right=.98,bottom=.10,top=.85,hspace=.54,wspace=1.4,height_ratios=[1.2,1,1])
    axes=[fig.add_subplot(gs[0,:6]),fig.add_subplot(gs[0,6:])]
    axes += [fig.add_subplot(gs[1,i*4:(i+1)*4]) for i in range(3)]
    axes += [fig.add_subplot(gs[2,i*3:(i+1)*3]) for i in range(4)]
    titles=["Overall held-out pass@1","Training reward · trailing 20 updates",*[f"{d.title()} tasks · {COUNTS[d]} per harness" for d in LEVELS],"OpenCode","Claude Code","Codex","Mini-SWE-Agent"]
    for ax,title in zip(axes,titles):
        ax.set_facecolor("white");ax.set_title(title,loc="left",pad=12);ax.grid(axis="y",alpha=.22)
        ax.spines[['top','right']].set_visible(False);ax.set_xlim(0,1000);ax.set_ylim(0,100 if ax==axes[1] else 80)
        ax.yaxis.set_major_formatter(PercentFormatter(100));ax.xaxis.set_major_locator(MultipleLocator(200));ax.set_xlabel("Optimizer step")
    for arm,run in snapshot["runs"].items():
        color=COLORS.get(arm,'#159a85');points=sorted(run["evaluations"].items());x=[s for s,_ in points]
        metrics=[score_metrics(v["score"],run["evaluations"][0]["score"]) for _,v in points]
        keys=["eval/pass_at_1",*[f"eval/difficulty/{d}/pass_at_1" for d in LEVELS],*[f"eval/harness/{h}/pass_at_1" for h in HARNESSES]]
        for ax,key in zip([axes[0],*axes[2:]],keys):
            ax.plot(x,[100*m[key] for m in metrics],color=color,marker='o',markersize=4,linewidth=2.3,label=arm)
        vals=[r.get("reward",r.get("rewards/harness_reward")) for r in run["training"]]
        avg=[sum(vals[max(0,i-19):i+1])/len(vals[max(0,i-19):i+1]) for i in range(len(vals))]
        axes[1].plot([r['step'] for r in run['training']],[100*v for v in avg],color=color,linewidth=1.6,alpha=.95)
        trained_points=[v for v in points if v[0]>0]
        if not trained_points:
            continue
        best=max(trained_points,key=lambda v:v[1]["score"]["average_pass_at_1"])
        bx=best[0];by=100*best[1]['score']['average_pass_at_1']
        axes[0].scatter([bx],[by],s=130,color=color,marker='*',zorder=5)
        axes[0].annotate(f"Best {by:.1f}% · step {bx}",(bx,by),xytext=(8,12 if arm==ARMS[0] else -22),textcoords='offset points',color=color,weight='bold',fontsize=10)
    axes[0].set_ylim(0,50)
    fig.text(.055,.96,"Qwen3.5-2B  /  Training comparisons",fontsize=25,weight='bold')
    fig.text(.055,.926,f"{len(snapshot['runs'])} runs · 1,000-step targets · recorded training and audited evaluations",fontsize=13,color="#596579")
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.98,.962),frameon=False,fontsize=12)
    when=snapshot['updated_utc'][:16].replace('T',' ')
    fig.text(.055,.05,f"AUDITED PASS@1  ·  250 fixed tests × 4 harnesses  ·  Updated {when} UTC",fontsize=10,weight='bold')
    fig.text(.055,.026,"Separate measured baseline cohorts (E2B / Daytona). Curves stop at the last audited checkpoint; pending evaluations are omitted. Full harness × difficulty tables accompany this figure.",fontsize=9,color="#596579")
    for suffix in ('png','svg','pdf'):
        fig.savefig(out/f"comparison.{suffix}",dpi=180,facecolor=fig.get_facecolor())
    plt.close(fig)


def sync(records, out):
    from dotenv import dotenv_values
    from trackio.remote_client import RemoteClient
    from trackio.sqlite_storage import SQLiteStorage
    values=dotenv_values(REPO/'experiments/.env')
    token=values.get('HF_API_KEY') or values['HF_TOKEN']
    client=RemoteClient(SPACE,hf_token=token,httpx_kwargs={'timeout':45})
    payload=SQLiteStorage.get_all_logs_for_sync(PROJECT)
    for start in range(0,len(payload),200):
        client.predict(api_name='/bulk_log',logs=payload[start:start+200],hf_token=token)
    configs=native.configuration_records(PROJECT)
    client.predict(api_name='/bulk_log',logs=configs,hf_token=token)
    proof=native.verify_remote_records(client,PROJECT,payload,timeout=120)
    for entry in configs:
        summary=client.predict(api_name='/get_run_summary',project=PROJECT,run_id=entry['run_id'])
        if summary.get('config')!=entry['config']:
            raise ValueError('Remote training configuration does not match source')
    proof['configurations_verified']=len(configs)
    native.write_json(out/'sync-receipt.json',{**proof,'checked_at':datetime.now(timezone.utc).isoformat(),'space':SPACE})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--online',action='store_true');p.add_argument('--watch',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    import tempfile,fcntl
    with (args.out/'.publisher.lock').open('w') as lock,tempfile.TemporaryDirectory(prefix='async-comparison-trackio-') as scratch:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for key in native.REMOTE_ENV:os.environ.pop(key,None)
        os.environ['TRACKIO_DIR']=scratch;os.environ['TRACKIO_STORAGE_MODE']='sqlite'
        last=None
        while True:
            try:
                snapshot=collect();records=events(snapshot);fingerprint=native.digest([[r['log_id'],r['config']] for r in records])
                native.import_events(records);native.backup_project(PROJECT,args.out/'trackio')
                if fingerprint!=last:
                    native.write_json(args.out/'snapshot.json',snapshot)
                    (args.out/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
                    report(snapshot,args.out);plot(snapshot,args.out)
                    if args.online:sync(records,args.out)
                    last=fingerprint
                state={'checked_at':datetime.now(timezone.utc).isoformat(),'ok':True,'events':len(records),
                       'training_steps':{a:len(r['training']) for a,r in snapshot['runs'].items()},
                       'evaluation_steps':{a:sorted(r['evaluations']) for a,r in snapshot['runs'].items()},'pending':snapshot['pending'],'online':args.online}
            except Exception as exc:
                state={'checked_at':datetime.now(timezone.utc).isoformat(),'ok':False,'error':str(exc)}
                if not args.watch:raise
            native.write_json(args.out/'publisher-status.json',state);print(json.dumps(state),flush=True)
            if not args.watch or (state.get('ok') and not state.get('pending')):break
            time.sleep(60)


if __name__=='__main__':main()
