"""Record qualification only after the student's real smoke artifacts pass."""
import argparse
import json
from pathlib import Path

from prepare import digest


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    arm = args.experiment.resolve()
    data, smoke = arm / 'data', arm / 'smoke'
    code = Path(__file__).resolve().parent
    preparation = read(data / 'preparation.json')
    steps = len(read(data / 'smoke_task_ids.json'))
    trained = read(smoke / f'training_verified_step_{steps}.json')
    assert trained['weights_changed'] and trained['native_trl_label_equality']
    assert trained['checkpoint_resume'] and trained['task_count'] == steps
    assert trained['max_sequence_length'] == preparation['lengths']['100']
    assert read(smoke / 'training_verified_step_2.json')['weights_changed']
    assert read(arm / 'epoch_eval_validation.json')['passed']
    assert read(arm / 'test_split_equivalence.json')['passed']
    assert read(arm / 'native_preparation_verified.json')['passed']
    evaluation = read(smoke / 'eval/evaluation_verified.json')
    assert evaluation['score']['complete'] and evaluation['score']['tito_pass']
    assert evaluation['score']['graded_cells'] == evaluation['score']['expected_cells'] == 8
    assert evaluation['model'] == preparation['model'] and evaluation['strictly_held_out']
    checkpoint = Path(trained['checkpoint'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    assert tokenizer.chat_template == (data / 'chat_template.jinja').read_text()
    logging = read(arm / 'local_logging_verified.json')
    assert logging['passed'] and logging['remote_requests'] == 0
    assert logging['training_steps_read_back'] == list(range(1, steps + 1))
    assert logging['eval_graded'] == 8
    proof = {'passed': True, 'model': preparation['model'], 'smoke_directory': str(smoke),
             'smoke_steps': steps, 'checkpoint_resume': True, 'longest_example_verified': True,
             'original_inference_template_restored': True, 'eval_graded': 8,
             'preparation': 'Full corpus verified without multiprocessing; GPU smoke used identical masks with one worker.',
             'load_test_at_100': False, 'train_sha256': digest(data / 'train.jsonl'),
             'scripts_sha256': {p.name: digest(p) for p in code.glob('*.py')}}
    (arm / 'qualification.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
