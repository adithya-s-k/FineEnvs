import run
from train_entry import LFMTokenizer


def test_existing_lfm_parser_handles_thinking_and_pythonic_tool_calls():
    tokenizer=LFMTokenizer.from_pretrained(run.C['model'],revision=run.C['model_revision'])
    parsed=tokenizer.parse_response("Check the file.</think><|tool_call_start|>[read_file(path='x.csv')]<|tool_call_end|><|im_end|>",
        prefix="<|im_start|>assistant\n<think>")
    assert parsed['tool_calls'][0]['function']=={'name':'read_file','arguments':{'path':'x.csv'}}
    assert parsed['thinking']=='Check the file.'
