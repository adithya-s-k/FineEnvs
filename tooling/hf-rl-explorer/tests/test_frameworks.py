"""The framework readers: NeMo Gym (app/envs/nemogym.py), verl / SkyRL (verl_rows.py) and Verifiers (verifiers_rows.py),
with the conversation and tool reader they share (convo.py). No network: rows are built here in each framework's real
field names, and the Prime Environments Hub and file listings are stubbed.

Checked first and most: no answer ever leaves a reader (not in the task view, its cards, the dataset overview, nor
what search looks at), for every framework's real answer fields. Then: each reader says what the policy is asked
(the conversation, its tools), how it's graded and by what, and how to run it with its own framework, honestly not
here.
"""

from __future__ import annotations

import json

import pytest

from app.envs import base, convo, nemogym, processors, rows, verifiers_rows, verl_rows
from app.envs.base import withhold


def ds(columns, sample, spec="org/ds", tags=(), splits=None):
    return base.Dataset(spec=spec, tags=list(tags), card={}, splits=splits or [{"config": "default", "split": "train"}],
                        features=[{"name": c, "type": {}} for c in columns], sample=sample)


def everything(proc, row, spec="org/ds", tags=(), splits=None):
    """A row through a reader the way the pages get it: its view, its card, and the dataset's overview."""
    d = ds(list(row), [row], spec=spec, tags=tags, splits=splits)
    roles = {**proc.roles(d), "_config": "default", "_split": "train"}
    view = proc.view(row, 0, d, roles)
    card = proc.card(row, 0, roles)
    overview = proc.overview(d, roles, [], "default", "train")
    return view, card, overview, roles, json.dumps([view, card, overview, {k: v for k, v in roles.items() if not k.startswith("_")}], ensure_ascii=False)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(verifiers_rows, "_hub", lambda env_id: {"runtime": "VERIFIERS_V1", "description": "a taskset"})
    monkeypatch.setattr(verifiers_rows, "_card_env", lambda spec, meta: None)


# ── conversations and tools ──────────────────────────────────────────────────
RESPONSES_INPUT = [
    {"role": "system", "content": "You are a finance assistant."},
    {"role": "user", "content": "Pull up the balance sheet for Suncor."},
    {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "I need the symbol."}]},
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Is **SU** right?", "annotations": []}]},
    {"role": "user", "content": "Yes, SU."},
    {"type": "function_call", "name": "get_balance_sheet", "arguments": "{\"symbol\": \"SU\"}", "call_id": "c1"},
    {"type": "function_call_output", "call_id": "c1", "output": "{\"assets\": 1}"},
    {"role": "user", "content": "Now the earnings, please."},
]
TOOLS = [{"type": "function", "name": "get_balance_sheet", "description": "Balance sheets. Annual and quarterly.", "strict": True,
          "parameters": {"type": "object", "properties": {"symbol": {"type": "string", "description": "Ticker"},
                                                          "period": {"type": "string", "enum": ["annual", "quarter"]}}, "required": ["symbol"]}},
         {"type": "function", "function": {"name": "get_earnings", "description": "EPS", "parameters": {"type": "object", "properties": {
             "symbol": {"type": "string"}, "years": {"type": "array", "items": {"type": "integer"}}}}}}]


def test_responses_items_become_a_transcript_with_outputs_paired_to_calls():
    ts = convo.turns(RESPONSES_INPUT)
    kinds = [(t["kind"], t["role"]) for t in ts]
    assert kinds == [("message", "system"), ("message", "user"), ("reasoning", "assistant"), ("message", "assistant"), ("message", "user"),
                     ("call", "assistant"), ("message", "user")]
    call = ts[5]
    assert call["name"] == "get_balance_sheet" and json.loads(call["args"]) == {"symbol": "SU"} and "\n" in call["args"]   # pretty-printed
    assert json.loads(call["output"]) == {"assets": 1}   # JSON outputs are pretty-printed too
    assert ts[3]["text"] == "Is **SU** right?" and ts[2]["text"] == "I need the symbol."
    assert convo.first_user(ts).startswith("Pull up") and convo.last_user(ts) == "Now the earnings, please."
    assert "get_balance_sheet" in convo.transcript_text(ts)


def test_chat_tool_calls_pair_with_tool_replies_in_order():
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "reasoning_content": "think", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "g", "arguments": "{\"x\": 1}"}}]},
            {"role": "tool", "tool_call_id": "b", "content": "from g"}, {"role": "tool", "content": "from f"}]
    ts = convo.turns(msgs)
    calls = [t for t in ts if t["kind"] == "call"]
    assert [c["output"] for c in calls] == ["from f", "from g"]
    assert ts[1]["kind"] == "reasoning" and not any(t["kind"] == "output" for t in ts)


def test_tools_read_as_signatures_whatever_their_shape():
    ts = convo.tools(TOOLS)
    assert [t["name"] for t in ts] == ["get_balance_sheet", "get_earnings"]
    sym, period = ts[0]["params"]
    assert sym == {"name": "symbol", "type": "string", "required": True, "description": "Ticker"}
    assert period["enum"] == ["annual", "quarter"] and not period["required"]
    assert ts[1]["params"][1]["type"] == "array of integer"
    assert convo.tools([{"name": "mcp_tool", "input_schema": {"properties": {"q": {"type": "string"}}, "required": ["q"]}}])[0]["params"][0]["required"]


def test_titles_skip_tags_templates_and_preambles():
    assert convo.task_title("<uploaded_files>\n/workspace/x\n</uploaded_files>\n\n<issue>\n**Title:** Mask ignored on assignment\n") == "Mask ignored on assignment"
    tpl = "Answer the following question. Reply with 'Answer: X'.\n\n{problem}"
    assert convo.task_title("Answer the following question. Reply with 'Answer: A/B'.\n\nWhat is 2+2?\nA: 4\nB: 5", tpl) == "What is 2+2?"
    assert convo.headline("Task:\nIn the sheet, sum column B.") == "In the sheet, sum column B."
    assert convo.headline("<image> As shown in the figure, find x.") == "As shown in the figure, find x."
    long = "We need a shortlist of films " + "with many constraints " * 12
    assert convo.headline(long).endswith("…") and len(convo.headline(long)) <= 182


def test_withhold_takes_dotted_names_through_lists():
    v, gone = withhold({"rubric": [{"question": "Q1?", "pass_criteria": "YES"}], "verifier_metadata": {"unit_tests": {"outputs": ["9"]}, "x": 1}},
                       hide=frozenset({"rubric.pass_criteria", "verifier_metadata.unit_tests"}))
    assert v == {"rubric": [{"question": "Q1?"}], "verifier_metadata": {"x": 1}}
    assert gone == ["rubric.pass_criteria", "verifier_metadata.unit_tests"]


# ── NeMo Gym ─────────────────────────────────────────────────────────────────
def pivot_row():
    return {"trajectory_id": 3344, "info": {"turn": 6, "step": 1, "depth": 12},
            "responses_create_params": {"input": RESPONSES_INPUT, "tools": TOOLS, "parallel_tool_calls": True},
            "expected_action": {"type": "function_call", "name": "get_earnings", "arguments": "{\"symbol\": \"SECRET-NEXT-STEP\"}"},
            "agent_ref": {"type": "responses_api_agents", "name": "toolcall_schema_single_step_tool_use_with_argument_comparison_agent"}}


def test_nemo_gym_pivot_row_reads_as_a_conversation_and_hides_the_experts_next_step():
    view, card, overview, roles, text = everything(nemogym.NemoGym(), pivot_row(), spec="nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1")
    assert "SECRET-NEXT-STEP" not in text
    assert view["withheld"] == ["expected_action"] and roles["answer"] == ["expected_action"]
    assert view["title"] == "Pull up the balance sheet for Suncor." and card["snippet"].startswith("Latest: Now the earnings")
    sec = {s["id"]: s for s in view["sections"]}
    tr = sec["task"]["body"][0]
    assert tr["type"] == "custom" and tr["renderer"] == "rl" and tr["kind"] == "transcript"
    assert any(t["kind"] == "call" and t.get("output") for t in tr["data"]["turns"]) and "expert" in tr["data"]["next"]["text"]
    assert sec["tools"]["body"][0]["kind"] == "tools" and len(sec["tools"]["body"][0]["data"]["tools"]) == 2
    grading = json.dumps(sec["grading"]["body"])
    assert "single_step_tool_use_with_argument_comparison" in grading and "expected_action" in grading
    run = sec["run"]["body"][0]["text"]
    assert "gym env start" in run and "resources_servers/single_step_tool_use_with_argument_comparison/configs/" \
                                      "toolcall_schema_single_step_tool_use_with_argument_comparison.yaml" in run
    assert "--agent toolcall_schema_single_step_tool_use_with_argument_comparison_agent" in run and "gym eval run --no-serve" in run
    assert "--repo-id nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1" in run and "sed -n '1p'" in run
    opt = view["framework_run"]
    # This browsing fixture mixes Chat Completions and Responses tool formats;
    # execution requires the dataset's native Responses function definitions.
    assert opt["runner"] == "nemo-gym" and opt["ok"] is False and "native NeMo Gym" in opt["why"]
    assert ["Step", "turn 6 · step 1"] in view["glance"] and card["chips"][0] == "turn 6 · step 1"


NEMO_ANSWERS = [
    # (row, the secret strings, the withheld names)
    ({"responses_create_params": {"input": [{"role": "user", "content": "Which? A: x B: y"}]}, "expected_answer": "SECRET-LETTER-C",
      "options": [{"A": "x"}, {"B": "y"}], "agent_ref": {"type": "responses_api_agents", "name": "mcqa_simple_agent"}},
     ["SECRET-LETTER-C"], ["expected_answer"]),
    ({"id": 0, "responses_create_params": {"input": [{"role": "user", "content": "Reply to carlos"}], "tools": TOOLS},
      "ground_truth": [{"name": "email_reply_email", "arguments": "{\"body\": \"SECRET-REPLY\"}"}], "category": "email",
      "environment_name": "workplace_assistant", "agent_ref": {"type": "responses_api_agents", "name": "workplace_assistant_simple_agent"}},
     ["SECRET-REPLY"], ["ground_truth"]),
    ({"responses_create_params": {"input": [{"role": "user", "content": "add a meeting"}]}, "exp_cal_state": {"0": {"min_time": "SECRET-10:00"}},
      "agent_ref": {"type": "responses_api_agents", "name": "calendar_simple_agent"}}, ["SECRET-10:00"], ["exp_cal_state"]),
    ({"responses_create_params": {"input": [{"role": "user", "content": "solve"}]}, "verifier_metadata": {"unit_tests": {"inputs": ["1"], "outputs": ["SECRET-OUT"]}},
      "hash_id": "h", "agent_ref": {"type": "responses_api_agents", "name": "code_gen_simple_agent"}}, ["SECRET-OUT"], ["verifier_metadata.unit_tests"]),
    ({"trajectory_id": 1, "responses_create_params": {"input": [{"role": "user", "content": "<issue>fix</issue>"}]},
      "ref_patch": "SECRET-PATCH", "ref_message": {"role": "assistant", "content": "SECRET-MSG"}, "expected_action": {"type": "message", "content": "SECRET-ACT"},
      "agent_ref": {"type": "responses_api_agents", "name": "single_step_tool_use_with_argument_comparison_swe"}},
     ["SECRET-PATCH", "SECRET-MSG", "SECRET-ACT"], ["expected_action", "ref_message", "ref_patch"]),
    ({"uuid": "u", "responses_create_params": {"input": [{"role": "user", "content": "plan a snack"}]},
      "rubric": [{"question": "Did it avoid peanuts?", "pass_criteria": "SECRET-YES"}], "agent_ref": {"type": "responses_api_agents", "name": "multichallenge_simple_agent"}},
     ["SECRET-YES"], ["rubric.pass_criteria"]),
    ({"task_id": "t", "responses_create_params": {"input": [{"role": "system", "content": "bank"}]}, "customer": "C1", "opening_message": "Hi, a statement please",
      "user_scenario": {"persona": "p"}, "initial_state": {}, "evaluation_criteria": {"actions": [{"name": "SECRET-ACTION"}], "communicate_info": ["SECRET-FACT"],
                                                                                   "nl_assertions": ["no mutation"]}},
     ["SECRET-ACTION", "SECRET-FACT"], ["evaluation_criteria.actions", "evaluation_criteria.communicate_info"]),
    ({"id": "j", "responses_create_params": {"input": [{"role": "user", "content": "how to"}]}, "response_policy_mapped": "SECRET-POLICY",
      "agent_ref": {"type": "responses_api_agents", "name": "jailbreak_detection_simple_agent"}}, ["SECRET-POLICY"], ["response_policy_mapped"]),
    ({"uuid": "r", "responses_create_params": {"input": [{"role": "user", "content": "knights"}]}, "question": "knights", "answer": "SECRET-ANS",
      "metadata": {"solution": "SECRET-SOLUTION"}, "agent_ref": {"type": "responses_api_agents", "name": "reasoning_gym_simple_agent"}},
     ["SECRET-ANS", "SECRET-SOLUTION"], ["answer", "metadata"]),
    ({"responses_create_params": {"input": [{"role": "user", "content": "cite"}]}, "verifier": {"type": "string_match", "patterns": ["SECRET-RE"],
                                                                                              "expected_markers": ["SECRET-MARK"]},
      "agent_ref": {"type": "responses_api_agents", "name": "citation_format_simple_agent"}}, ["SECRET-RE", "SECRET-MARK"], ["verifier"]),
]


@pytest.mark.parametrize(("row", "secrets", "names"), NEMO_ANSWERS)
def test_nemo_gym_answer_fields_never_leave(row, secrets, names):
    view, card, overview, roles, text = everything(nemogym.NemoGym(), row)
    for s in secrets:
        assert s not in text, s
    assert set(names) <= set(view["withheld"]), view["withheld"]


def test_nemo_gym_names_the_config_that_runs_a_renamed_or_unnamed_agent():
    a = nemogym.agent_of({"agent_ref": {"type": "responses_api_agents", "name": "turing_vif_simple_agent"}}, "org/x")
    assert a["known"] and a["renamed"] and a["config_name"] == "verifif_simple_agent" and a["resources_server"] == "verifif"
    a = nemogym.agent_of({"agent_ref": {"name": "single_step_tool_use_with_argument_comparison_swe"}}, "org/x")
    assert a["config_name"] == "swe_pivot_single_step_tool_use_with_argument_comparison_agent"
    a = nemogym.agent_of({"task_id": "t", "customer": "C", "user_scenario": {}, "evaluation_criteria": {}, "initial_state": {}, "opening_message": "hi"}, "org/x")
    assert a["inferred"] and a["resources_server"] == "indian_banking"
    a = nemogym.agent_of({"question": "q", "expected_answer": "1"}, "nvidia/Nemotron-RL-math-OpenMathReasoning")   # no agent_ref: the config naming the dataset
    assert a["resources_server"] == "math_with_judge"
    a = nemogym.agent_of({"agent_ref": {"name": "nobody_knows_agent"}}, "org/x")
    assert not a["known"] and a["name"] == "nobody_knows_agent"


def test_nemo_gym_blends_route_each_row_and_run_without_one_agent():
    rows_ = [{"responses_create_params": {"input": [{"role": "user", "content": f"q{i}"}]},
              "agent_ref": {"type": "responses_api_agents", "name": n}} for i, n in enumerate(["mcqa_simple_agent", "workplace_assistant_simple_agent"])]
    d = ds(["responses_create_params", "agent_ref"], rows_, spec="nvidia/Nemotron-RL-Ultra-Training-Blends")
    p = nemogym.NemoGym()
    roles = p.roles(d)
    assert roles["_mixed"] and p.card(rows_[0], 0, roles)["lead"] == "mcqa_simple_agent"
    ov = p.overview(d, roles, [], "rlvr1", "train")
    run = next(s for s in ov if s["id"] == "run")["blocks"][0]["text"]
    assert run.count("--config ") == 2 and "--agent" not in run and "load_dataset('nvidia/Nemotron-RL-Ultra-Training-Blends', 'rlvr1'" in run


def test_nemo_gym_catalog_is_current_and_complete():
    cat = nemogym.catalog()
    assert len(cat["commit"]) == 40 and len(cat["agents"]) > 100 and len(cat["servers"]) > 100
    assert cat["servers"]["instruction_following"]["verification"] == "Rule-based instruction checks"
    assert cat["servers"]["math_proof_judgement"]["verification"] == "Deterministic answer parsing"
    for name in ("mcqa_simple_agent", "workplace_assistant_simple_agent", "calendar_simple_agent", "code_gen_simple_agent",
                 "instruction_following_simple_agent", "reasoning_gym_simple_agent"):
        a = cat["agents"][name]
        assert a["config"].startswith("resources_servers/") and a["resources_server"] in cat["servers"]


# ── verl / SkyRL ─────────────────────────────────────────────────────────────
def verl_row(**kw):
    return {"data_source": "openai/gsm8k", "prompt": [{"role": "user", "content": "Natalia sold clips to 48 friends. How many? Let's think step by step."}],
            "ability": "math", "reward_model": {"style": "rule", "ground_truth": "SECRET-72"},
            "extra_info": {"split": "train", "index": 0, "answer": "SECRET-WORKED-SOLUTION", "question": "Natalia sold clips to 48 friends. How many?",
                           "tools_kwargs": {"calc_gsm8k_reward": {"create_kwargs": {"ground_truth": "SECRET-TOOL-GT"}}}}, **kw}


def test_verl_rows_hide_every_ground_truth_and_name_their_scorer():
    view, card, overview, roles, text = everything(verl_rows.Verl(), verl_row(agent_name="tool_agent"))
    for s in ("SECRET-72", "SECRET-WORKED-SOLUTION", "SECRET-TOOL-GT"):
        assert s not in text, s
    assert {"reward_model.ground_truth", "extra_info.answer"} <= set(view["withheld"])
    assert view["title"] == "Natalia sold clips to 48 friends. How many?" and ["Scorer", "`gsm8k`"] in view["glance"]
    sec = {s["id"]: s for s in view["sections"]}
    assert "calc_gsm8k_reward" in json.dumps(sec["tools"]["body"])
    run = sec["run"]["body"][0]["text"]
    assert "python3 -m verl.trainer.main_ppo" in run and "multi_turn.enable=True" in run and "custom_reward_function" not in run
    assert view["framework_run"]["runner"] == "verl" and view["framework_run"]["ok"] is False


@pytest.mark.parametrize(("source", "scorer"), [("openai/gsm8k", "gsm8k"), ("HuggingFaceH4/MATH-500", "math_reward"), ("aime24", "math_dapo"),
                                                ("numina_olympiads", "prime_math"), ("codeforces", "sandbox_fusion / prime_code"),
                                                ("searchR1_nq", "search_r1_like_qa_em"), ("DeepScaleR-Preview", None)])
def test_verl_routes_data_sources_as_default_compute_score_does(source, scorer):
    got = verl_rows.scorer(source)
    assert (got[0] if got else None) == scorer


def test_verl_rows_with_no_built_in_scorer_get_a_custom_reward_in_the_command():
    view, *_ = everything(verl_rows.Verl(), {**verl_row(), "data_source": "sheet_arena"})
    run = next(s for s in view["sections"] if s["id"] == "run")["body"][0]["text"]
    assert "reward.custom_reward_function.path=" in run


def test_compressed_blobs_and_skyrl_sql_answers_are_withheld():
    blob = "H4sI" + "A" * 400
    row = {"data_source": "synsql", "prompt": [{"role": "user", "content": "Write SQL for the question"}], "env_class": "text2sql",
           "reward_spec": {"method": "rule", "ground_truth": "SECRET-GT"}, "sql": "SELECT SECRET_SQL", "output_seq": "SECRET-REASONING",
           "db_id": "shop", "extra_info": {"part_1": blob}}
    view, card, overview, roles, text = everything(verl_rows.Verl(), row)
    for s in ("SECRET-GT", "SECRET_SQL", "SECRET-REASONING", blob):
        assert s not in text
    assert view["framework"] == "SkyRL" and "extra_info.part_1" in view["withheld"]
    run = next(s for s in view["sections"] if s["id"] == "run")["body"][0]["text"]
    assert "skyrl.train.entrypoints.main_base" in run and "environment.env_class=text2sql" in run and "org/ds:train" in run


def test_mimo_no_longer_claims_verl_rows_that_name_an_agent_loop():
    row = {"data_source": "sheet_arena", "agent_name": "tool_agent", "prompt": [{"role": "user", "content": "x"}], "ability": "spreadsheet",
           "reward_model": {"style": "rule", "ground_truth": "id"}, "extra_info": {"id": "x"}}
    assert rows.pick(ds(list(row), [row])).id == "verl"
    mimo = {**row, "extra_info": {"instance_json": "{}"}}
    assert rows.pick(ds(list(mimo), [mimo])).id == "mimo"


# ── Verifiers ────────────────────────────────────────────────────────────────
def test_verifiers_swe_rows_withhold_gold_patches_tests_and_solutions():
    row = {"repo_name": "orange3", "docker_image": "namanjain12/orange3:abc", "commit_hash": "abc",
           "problem_statement": "[ISSUE]\n**Title:** Context migration fails\n\n**Description:** It raises.\n[/ISSUE]",
           "prompt": "Write an issue for this commit SECRET-COMMIT-PROMPT", "parsed_commit_content": "SECRET-GOLD-DIFF",
           "execution_result_content": "SECRET-TESTS", "expected_output_json": "{\"test_a\": \"SECRET-PASSED\"}", "modified_files": ["SECRET/file.py"],
           "relevant_files": ["SECRET/rel.py"], "modified_entity_summaries": "SECRET-SUMMARY"}
    view, card, overview, roles, text = everything(verifiers_rows.Verifiers(), row, spec="PrimeIntellect/R2E-Gym-Subset-Verified")
    for s in ("SECRET-COMMIT-PROMPT", "SECRET-GOLD-DIFF", "SECRET-TESTS", "SECRET-PASSED", "SECRET/file.py", "SECRET/rel.py", "SECRET-SUMMARY"):
        assert s not in text, s
    assert view["title"] == "Context migration fails"
    assert ["Image", "`namanjain12/orange3:abc`"] in view["glance"] and ["API", "v1"] in view["glance"]
    run = next(s for s in view["sections"] if s["id"] == "run")["body"][0]["text"]
    assert "prime env install primeintellect/r2e-gym" in run and "uv run eval primeintellect/r2e-gym" in run and "eval.toml" not in run


def test_multi_swe_bench_shows_the_issues_and_hides_the_fixing_pr():
    row = {"org": "facebook", "repo": "zstd", "number": 3942, "title": "Fix #3719: SECRET-PR-TITLE", "body": "SECRET-PR-BODY",
           "resolved_issues": {"title": ["zstd won't remove the file"], "body": ["Steps to reproduce"], "number": [3719]},
           "fix_patch": "SECRET-FIX", "test_patch": "SECRET-TEST-PATCH", "f2p_tests": {"t": "SECRET-F2P"}, "run_result": {"x": "SECRET-RUN"},
           "instance_id": "facebook__zstd-3942", "hints": "SECRET-HINT"}
    view, card, overview, roles, text = everything(verifiers_rows.Verifiers(), row, spec="PrimeIntellect/Multi-SWE-bench",
                                                   splits=[{"config": "default", "split": "test"}])
    for s in ("SECRET-PR-TITLE", "SECRET-PR-BODY", "SECRET-FIX", "SECRET-TEST-PATCH", "SECRET-F2P", "SECRET-RUN", "SECRET-HINT"):
        assert s not in text, s
    assert view["title"] == "zstd won't remove the file" and roles["task"] == "resolved_issues"
    run = next(s for s in view["sections"] if s["id"] == "run")["body"][0]["text"]
    assert 'dataset_name = "PrimeIntellect/Multi-SWE-bench"' in run and 'id = "primeintellect/multiswe"' in run   # not the taskset's default dataset


def test_verifiers_v0_rows_show_the_prompt_and_withhold_the_answer():
    row = {"question": "What is 6*7?", "answer": "SECRET-42", "info": {"difficulty": "easy", "solution": "SECRET-STEPS"}, "task": "arith"}
    view, card, overview, roles, text = everything(verifiers_rows.Verifiers(), row, tags=["verifiers"])
    assert "SECRET-42" not in text and "SECRET-STEPS" not in text
    assert view["title"] == "What is 6*7?" and {"answer", "info.solution"} <= set(view["withheld"])
    assert "difficulty" in json.dumps(view["sections"])
    run = next(s for s in view["sections"] if s["id"] == "run")["body"][0]["text"]
    assert "prime env install <owner>/<environment>" in run   # no environment named: says how to find it, invents none


def test_verifiers_known_v0_environment_uses_the_legacy_runner(monkeypatch):
    monkeypatch.setattr(verifiers_rows, "_hub", lambda env_id: {"runtime": "VERIFIERS_V0"})
    view, *_ = everything(verifiers_rows.Verifiers(), {"prompt": "The community in Bruck was merged into it"}, spec="PrimeIntellect/Reverse-Text-RL")
    task = next(s for s in view["sections"] if s["id"] == "task")["body"][0]["data"]["turns"]
    assert task[0]["role"] == "system" and "<reversed_text>" in task[0]["text"]
    run = next(s for s in view["sections"] if s["id"] == "run")["body"][0]["text"]
    assert "uv run vf-eval reverse-text" in run and "prime env install primeintellect/reverse-text" in run


def test_rule_induction_rows_hide_the_rule():
    row = {"rule_id": "red_cards", "label": "red cards", "family": "single_axis", "code": "return card.color == 'red'"}
    view, card, overview, roles, text = everything(verifiers_rows.Verifiers(), row, spec="nph4rd/eleusis-simple-rules", tags=["verifiers"])
    assert "red_cards" not in text and "red cards" not in text and "card.color" not in text


# ── the rows adapter ─────────────────────────────────────────────────────────
def test_search_never_looks_at_framework_answers():
    row = {"expected_action": {"content": "SECRET"}, "ref_patch": "SECRET", "exp_cal_state": "SECRET", "fix_patch": "SECRET", "prompt": "find me"}
    assert "SECRET" not in json.dumps(rows._searchable(row)) and "find me" in json.dumps(rows._searchable(row))


def test_framework_rows_offer_their_own_framework_as_the_way_to_run():
    opt = nemogym.run_option({"resources_server": "mcqa"})
    assert rows.RowsAdapter().run_options(None, "train/0", {"_framework_run": opt}) == [opt]
    assert opt["ok"] is False and opt["why"] and "**Run it with NeMo Gym**" in opt["about"]


def test_a_dataset_opens_on_its_train_split_not_its_leftovers():
    assert rows._first([{"config": "default", "split": "dropped"}, {"config": "default", "split": "resolved"}])["split"] == "resolved"
    assert rows._first([{"config": "a", "split": "test"}, {"config": "a", "split": "train"}, {"config": "b", "split": "train"}]) == {"config": "a", "split": "train"}


def test_processors_hidden_union_covers_every_framework():
    assert {"expected_action", "sql", "fix_patch"} <= processors.HIDDEN


def test_a_config_the_viewer_cant_read_is_read_from_its_files(monkeypatch):
    class BrokenViewer:
        kind, tok = "viewer", None

        def page(self, config, split, offset, length):
            raise rows.ViewerError("The dataset generation failed", 502)

    class FakeFiles:
        kind = "files"

        def __init__(self, spec, meta, tok):
            pass

        def splits(self):
            return [{"config": "rlvr1", "split": "train"}]

        def page(self, config, split, offset, length):
            return {"rows": [(0, {"responses_create_params": {"input": "hi"}}, [])], "total": 1, "features": []}

    monkeypatch.setattr(rows, "Files", FakeFiles)
    rows._split_files.clear()
    be, page = rows._first_page("org/blend", {"sha": "abc"}, BrokenViewer(), {"config": "rlvr1", "split": "train"})
    assert be.kind == "files" and page["total"] == 1
    be, _ = rows._first_page("org/blend", {"sha": "abc"}, BrokenViewer(), {"config": "rlvr1", "split": "train"})   # remembered
    assert be.kind == "files"
    with pytest.raises(rows.ViewerError):   # a split the files don't have either: the viewer's error stands
        rows._first_page("org/blend", {"sha": "abc"}, BrokenViewer(), {"config": "other", "split": "train"})


def test_blend_rows_are_pivots_only_when_they_carry_an_expert_step_and_placeholders_say_where_from():
    plain = {"responses_create_params": {"input": [{"role": "user", "content": "q"}]}, "expected_action": None, "ref_message": None,
             "agent_ref": {"type": "responses_api_agents", "name": "mcqa_simple_agent"}}
    assert not nemogym.is_pivot(plain) and nemogym.is_pivot(pivot_row())
    ph = {"question": "", "responses_create_params": {"input": [{"role": "user", "content": ""}]}, "expected_answer": "",
          "agent_ref": {"type": "responses_api_agents", "name": "math_with_judge_simple_agent"},
          "_hf_question_placeholder": {"mode": "canonical", "dataset": "Skywork/Skywork-OR1-RL-Data", "split": "math", "row": 246}}
    view, card, *_ = everything(nemogym.NemoGym(), ph)
    assert card["title"] == "A question from Skywork/Skywork-OR1-RL-Data (math, row 246)"
    assert "fills it in from" in json.dumps(view["sections"][0]["body"])


def test_search_never_finds_a_row_through_its_withheld_answer(monkeypatch):
    """The dataset viewer's search matches every column; a hit found only through an answer is dropped, and the total
    then says nothing about how many rows hold it. Rows read from files are searched without their answers too."""
    from app.envs import rows, viewer

    found = [(0, {"prompt": "add 2 and 3", "reward_model": {"ground_truth": "5"}}, []),
             (1, {"prompt": "the answer is 5 apples", "reward_model": {"ground_truth": "7"}}, [])]
    monkeypatch.setattr(viewer, "search", lambda *a, **k: {"rows": found, "total": 900, "features": []})
    rows._broken.clear()
    page, _ = rows.Viewer("o/ds", None).search("default", "train", "5", 0)
    assert [i for i, _, _ in page["rows"]] == [1] and page["total"] == 1
    page, _ = rows.Viewer("o/ds", None).search("default", "train", "apple", 0)   # a stemmed word still counts as visible
    assert [i for i, _, _ in page["rows"]] == [1]

    files = rows.Files.__new__(rows.Files)
    monkeypatch.setattr(rows.Files, "_scan", lambda self, c, s, o, match, what: ({"rows": [h for h in found if match(h[1])]}, None))
    assert [i for i, _, _ in files.search("d", "train", "7", 0)[0]["rows"]] == []


def test_verifiers_v1_taskdata_system_prompt_is_part_of_the_conversation():
    row = {"prompt": "Solve the task", "system_prompt": "Use the supplied tools", "answer": "PRIVATE_REFERENCE"}
    view, _, _, _, text = everything(verifiers_rows.Verifiers(), row, tags=["verifiers"])
    turns = next(s for s in view["sections"] if s["id"] == "task")["body"][0]["data"]["turns"]
    assert [(t["role"], t["text"]) for t in turns] == [("system", "Use the supplied tools"), ("user", "Solve the task")]
    assert "PRIVATE_REFERENCE" not in text
