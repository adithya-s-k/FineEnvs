"""Run delayed fixture completions through a real relay and exact-token capture."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import time

import httpx

from openenv.core.harness.capture.export import export_session
from openenv.core.harness.capture.forwarding import GradioForwarder
from openenv.core.harness.capture.runner import CaptureServer


class FixtureEngine:
    served_model = 'fixture-model'
    param_fixes = {}
    api_key = None
    calls = 0

    async def completion(self, request):
        self.calls += 1
        assert request['stream'] is False
        slow = 'second' in json.dumps(request['messages'])
        await asyncio.sleep(75 if slow else .01)
        ids = [6, 7] if slow else [3, 4]
        return {'id': 'fixture-second' if slow else 'fixture-first', 'object': 'chat.completion',
                'model': self.served_model, 'prompt_token_ids': [1, 2, 3, 4, 5] if slow else [1, 2],
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'fixture answer'},
                'finish_reason': 'stop', 'token_ids': ids,
                'logprobs': {'content': [{'token': str(i), 'logprob': -.25} for i in ids]}}],
                'usage': {'prompt_tokens': 5 if slow else 2, 'completion_tokens': 2}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    server = CaptureServer(llm_url='http://127.0.0.1:9/v1', model='fixture-model', port=port,
                           capture_level='tokens')
    engine = FixtureEngine()
    server.app.state.inference = engine
    server.app.state.upstreams._default = (engine, 'tokens')
    forwarder = GradioForwarder()
    try:
        server.start()
        url = forwarder.start(port)
        def probe(dialect):
            session = server.app.state.registry.create(max_model_calls=2)
            receipts = []
            for prompt in ['first', 'second']:
                body = {'model': 'fixture-model', 'stream': True,
                        'messages': [{'role': 'user', 'content': prompt}]}
                path = '/v1/chat/completions'
                if dialect == 'anthropic':
                    path = '/v1/messages'; body['max_tokens'] = 10
                elif dialect == 'responses':
                    path = '/v1/responses'; body.pop('messages'); body['input'] = prompt
                elif dialect == 'google':
                    path = '/v1beta/models/fixture-model:streamGenerateContent?alt=sse'
                    body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}]}
                began = time.monotonic()
                first_byte, heartbeats, data = None, 0, []
                with httpx.stream('POST', url + path, json=body, timeout=100,
                    headers={'Authorization': 'Bearer ' + session.session_id}) as response:
                    assert response.status_code == 200, response.status_code
                    for line in response.iter_lines():
                        if first_byte is None:
                            first_byte = time.monotonic() - began
                        heartbeats += line.startswith(': openenv keepalive')
                        if line.startswith('data: ') and line != 'data: [DONE]':
                            data.append(json.loads(line[6:]))
                assert 'fixture answer' in json.dumps(data), dialect
                receipts.append({'request': prompt, 'seconds': time.monotonic() - began,
                                 'first_byte_seconds': first_byte, 'heartbeats': heartbeats})
            document = export_session(session, capture_level='tokens')
            assert len(document['turns']) == session.model_calls == 2
            assert len(document['sequences']) == 1
            sequence = document['sequences'][0]
            assert sequence['input_ids'] == [1, 2, 3, 4, 5, 6, 7]
            assert sequence['loss_mask'] == [0, 0, 1, 1, 0, 1, 1]
            assert sequence['logprobs'] == [0, 0, -.25, -.25, 0, -.25, -.25]
            assert receipts[1]['heartbeats'] >= 6 and receipts[1]['first_byte_seconds'] < 20
            return {'dialect': dialect, 'passed': True, 'receipts': receipts, 'captured_turns': 2,
                    'supervised_tokens': 4}
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(probe, ['chat', 'anthropic', 'responses', 'google']))
        assert engine.calls == 8
        report = {'passed': True, 'fixture_only_no_model_or_sandbox': True,
                  'engine_calls': engine.calls, 'results': results}
        (args.out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    finally:
        forwarder.stop()
        server.stop()


if __name__ == '__main__':
    main()
