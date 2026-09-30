"""Verify current-turn supervision with the actual pinned student tokenizer."""
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer
from prepare import MODEL, MODEL_REVISION
from prepare_opencode import normalize_messages, student_tokens

tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
tools = [{'type': 'function', 'function': {'name': 'bash', 'description': 'Run a command',
          'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}]
prompt = [{'role': 'system', 'content': 'Use the tools.'}, {'role': 'user', 'content': 'Compute 2+2.'},
          {'role': 'assistant', 'content': 'Earlier assistant text must be context only.'},
          {'role': 'tool', 'tool_call_id': 'previous', 'content': 'Earlier tool result.'}]
completion = normalize_messages([{'role': 'assistant', 'content': None,
              'tool_calls': [{'id': 'now', 'type': 'function',
                              'function': {'name': 'bash', 'arguments': '{"command":"echo 4"}'}}]}])
record, supervised = student_tokens(tokenizer, prompt, completion, tools)
target = tokenizer.decode([x for x in record['labels'] if x != -100])
assert 'Earlier assistant' not in target and 'Earlier tool' not in target
assert "bash(command='echo 4')" in target and '<|im_end|>' in target
assert len(record['input_ids']) > supervised > 0
assert all(label == -100 for label in record['labels'][:-supervised])
assert '<think>' not in target
record2, _ = student_tokens(tokenizer, prompt, [{'role': 'assistant', 'content': '4', 'reasoning': 'Two plus two.'}], tools)
target2 = tokenizer.decode([x for x in record2['labels'] if x != -100])
assert '<think>Two plus two.</think>4' in target2
try:
    normalize_messages([{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'not-used'}}]}])
except AssertionError:
    pass
else:
    raise AssertionError('Unsupported multimodal input was silently accepted')
proof = {'passed': True, 'prompt_assistants_and_tool_outputs_masked': True,
         'current_tool_call_and_eos_supervised': True, 'reasoning_preserved_when_present': True,
         'no_synthetic_reasoning_inserted': True, 'unsupported_multimodal_fails_closed': True}
Path(sys.argv[1]).write_text(json.dumps(proof, indent=2) + '\n')
print(json.dumps(proof))
