"""Replay local scalar logs to the existing comparison dashboard off the training path."""
import argparse
import json
import os
from pathlib import Path
import time


def sync(output, config):
    output = Path(output)
    os.environ['TRACKIO_DIR'] = str(output / 'dashboard-db')
    os.environ['TRACKIO_STORAGE_MODE'] = 'sqlite'
    import trackio_multi4 as native
    from trackio.remote_client import RemoteClient
    from trackio.sqlite_storage import SQLiteStorage
    project = config['logging']['project'] + ('-smoke' if os.environ.get('HF_PHASE') == 'smoke' else '')
    name = os.environ['RUN_OWNER'] if os.environ.get('HF_PHASE') == 'smoke' else config['run_name']
    metadata = {'parent_step':500,'hard_tasks':500,'task_epochs':2,'bundle_sha256':os.environ['BUNDLE_SHA256']}
    records = []
    metrics = output / 'audit/metrics.jsonl'
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(native.event(project, name, row['step'], native.scalars(
                {k:v for k,v in row.items() if k != 'step'}, 'train/'), metadata))
    score_path = output / 'canonical_scores.json'
    if score_path.exists():
        score = json.loads(score_path.read_text())
        if score.get('comparison_ready'):
            values = {'eval/pass_at_1':score['average_pass_at_1']}
            for h, item in score['harnesses'].items():
                values[f'eval/{h}/pass_at_1'] = item['pass_at_1']
                for d, detail in item['difficulty'].items():
                    if detail['pass_at_1'] is not None:
                        values[f'eval/{h}/{d}/pass_at_1'] = detail['pass_at_1']
            records.append(native.event(project, name, int(os.environ.get('CHECKPOINT_STEP','0')),values,metadata))
    if not records:
        return
    native.import_events(records)
    native.backup_project(project, output / 'trackio-backup')
    client = RemoteClient(config['logging']['space_id'], hf_token=os.environ['HF_TOKEN'], httpx_kwargs={'timeout':30})
    logs = SQLiteStorage.get_all_logs_for_sync(project)
    for i in range(0,len(logs),500):
        client.predict(api_name='/bulk_log',logs=logs[i:i+500],hf_token=os.environ['HF_TOKEN'])
    native.verify_remote_records(client,project,logs)
    (output/'trackio_verified.json').write_text(json.dumps({'updated_at':time.time(),'project':project,
        'run':name,'events':len(logs),'remote_readback_verified':True})+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    while True:
        try:
            sync(args.output,config)
        except Exception as exc:
            (args.output/'trackio_error.json').write_text(json.dumps({'time':time.time(),'error_type':type(exc).__name__})+'\n')
        if not args.watch:
            return
        time.sleep(60)


if __name__=='__main__':
    main()
