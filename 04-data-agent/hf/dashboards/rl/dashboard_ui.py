"""Use Trackio's supported custom frontend with focused, shareable starting views."""
import json
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlencode

PROJECT = 'data-agent-rl-comparison'
LFM = '2710d85285055cff752a1ab35e0ac343,7a4d3a85b9b0718a035f6242627ebfc7'
QWEN = '6d2af97e3918de0c8781b41b90d412dd,7419b3527b9fee9206780cbc9ac20983,8feb4c3f6574d8f228c60d97ece084db'
PATTERN = r'^(eval_observed/(pass_at_1|combined_reward_normalized|tool_call_savings_pct)|train_verified/reward_mean50|eval_coverage/(graded_cells|missing_cells))$'


def url(run_ids, pattern, **extra):
    return '/?' + urlencode(dict(project=PROJECT, run_ids=run_ids, metric_filter=pattern, smoothing=0, accordion='hidden', **extra))


def prepare_frontend():
    from trackio.frontend_config import BUNDLED_FRONTEND_DIR
    target = Path(tempfile.mkdtemp(prefix='rl-trackio-frontend-'))
    shutil.copytree(BUNDLED_FRONTEND_DIR, target, dirs_exist_ok=True)
    defaults = dict(project=PROJECT, run_ids=LFM, metric_filter=PATTERN, smoothing='0', accordion='hidden')
    script = '''<script>
    (() => {
      const url = new URL(window.location.href);
      if ((url.pathname === '/' || url.pathname === '/metrics') &&
          !['metric_filter', 'metrics', 'run_ids', 'runs', 'view'].some(k => url.searchParams.has(k))) {
        const defaults = DEFAULTS;
        for (const [k, v] of Object.entries(defaults)) url.searchParams.set(k, v);
        history.replaceState(history.state, '', url);
      }
    })();
    </script>'''.replace('DEFAULTS', json.dumps(defaults))
    links = [
        ('LFM overview', url(LFM, PATTERN)),
        ('Qwen overview', url(QWEN, r'^(eval/(pass_at_1|provisional_pass_at_1|graded_cells)|train/reward_rolling20)$')),
        ('Harness scores', url(LFM, r'^eval_observed/harness/')),
        ('Tool and token usage', url(LFM, r'^eval_usage/(overall|harness/[^/]+)/(generated_tokens_mean|input_tokens_mean|native_tool_calls_mean)$')),
        ('All RL metrics', '/?project='+PROJECT+'&view=all'),
        ('SmolDataEnv SFT', 'https://huggingface.co/spaces/FineEnvs/data-agent-sft-trackio'),
    ]
    import html
    navigation = '<nav id="rl-views" aria-label="RL dashboard views"><strong>SmolDataEnv RL</strong>'
    navigation += ''.join(f'<a href="{html.escape(href, quote=True)}">{label}</a>' for label, href in links)
    navigation += '</nav>'
    styles = '''<style>
    #rl-views {box-sizing:border-box;display:flex;align-items:center;gap:18px;min-height:48px;padding:10px 16px;background:#111827;color:#e5e7eb;font:13px system-ui;overflow-x:auto;white-space:nowrap;border-bottom:1px solid #374151}
    #rl-views a {color:#93c5fd;text-decoration:none} #rl-views a:hover{text-decoration:underline}
    #app > .app {height:calc(100vh - 48px)}
    </style>'''
    index = target/'index.html'
    source = index.read_text().replace('<title>Trackio Dashboard</title>', '<title>SmolDataEnv RL</title>').replace('<head>', '<head>\n'+script+styles).replace('<body>', '<body>\n'+navigation)
    index.write_text(source)
    return target
