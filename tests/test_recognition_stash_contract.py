"""HTTP contract regression for Stash v0.31.1 BulkUpdateIds.ids (not values).

Source: https://github.com/stashapp/stash/blob/v0.31.1/graphql/schema/types/scene.graphql
The endpoint enforces that upstream input shape before any association write.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
from types import SimpleNamespace

import pytest
import requests

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.publication import LINK_INPUT_CONTRACT, PendingRecognition, publish_person, save_operation
from stash_ai.recognition.stash import RecognitionStash
from stash_ai.runtime.report import RunReport


@contextmanager
def endpoint():
    state = SimpleNamespace(calls=[], performers=['77'], rejection=None, message='schema rejected input')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            assert self.path == '/graphql' and self.headers['ApiKey'] == 'private-key'
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state.calls.append(request)
            query, variables = request['query'], request['variables']
            status = 200
            if 'StashAIVersion' in query:
                result = {'version': {'version': 'v0.31.1'}}
            elif 'StashAIRecognitionPerformer' in query:
                result = {'findPerformer': {'id': '5', 'name': 'AI Person', 'details': 'owned-token', 'image_path': None}}
            elif 'StashAIRecognitionScene' in query:
                result = {'findScene': {'id': '1', 'performers': [{'id': p} for p in state.performers]}}
            elif 'StashAIRecognitionLink' in query:
                association = variables['input']['performer_ids']
                if set(association) != {'ids', 'mode'}:
                    status, result = 400, {'errors': [{'message': 'Field "values" is not defined by type "BulkUpdateIds".'}]}
                elif state.rejection:
                    status, result = 400, {'errors': [{'message': state.message}]}
                    if state.rejection == 'validation':
                        result['errors'][0]['extensions'] = {'code': 'GRAPHQL_VALIDATION_FAILED'}
                else:
                    assert variables['input']['ids'] == ['1'] and association['mode'] == 'ADD'
                    state.performers = list(dict.fromkeys([*state.performers, *association['ids']]))
                    result = {'bulkSceneUpdate': [{'id': '1', 'performers': [{'id': p} for p in state.performers]}]}
            else:
                raise AssertionError(query)
            body = json.dumps(result if status == 400 else {'data': result}).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
    thread.start()
    client = None
    try:
        client = RecognitionStash(url=f'http://127.0.0.1:{server.server_port}', api_key='private-key', timeout=3)
        yield client, state
    finally:
        if client:
            client.close()
        server.shutdown()
        thread.join()
        server.server_close()


def links(state):
    return [call for call in state.calls if 'StashAIRecognitionLink' in call['query']]


def person(db):
    with db.connection:
        db.connection.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',
                              ('person', 'scope', 'model', 0, '5', 'owned-token'))


def test_legacy_shape_reproduces_400_and_correct_request_preserves_manual_performers():
    with endpoint() as (client, state):
        with pytest.raises(requests.HTTPError, match='values.*BulkUpdateIds') as caught:
            client.gql('mutation StashAIRecognitionLink($input:BulkSceneUpdateInput!) {'
                       'bulkSceneUpdate(input:$input) { id performers { id } }}',
                       {'input': {'ids': ['1'], 'performer_ids': {'values': ['5'], 'mode': 'ADD'}}})
        assert caught.value.response.status_code == 400 and state.performers == ['77']
        assert len(links(state)) == 1  # A rejected mutation is never automatically retried.
        client.add_scene_person('1', '5')
        assert state.performers == ['77', '5']
        assert links(state)[-1]['variables']['input']['performer_ids'] == {'ids': ['5'], 'mode': 'ADD'}


def test_original_400_journal_recovers_on_explicit_run_and_retains_evidence(tmp_path):
    with endpoint() as (client, state), FaceDatabase(tmp_path / 'faces.sqlite3') as db:
        person(db)
        save_operation(db, scope='scope', scene='1', person='person', kind='link', target='5',
                       payload={'performer': '5'}, precondition={'performers': ['77']}, status='unknown',
                       error=f'400 Client Error: Bad Request for url: {client.url}/graphql')
        report = RunReport(write=lambda _: None)
        assert publish_person(db, client, report, 'scope', '1', 'person') == '5'
        row = db.connection.execute("SELECT * FROM publications WHERE kind='link'").fetchone()
        assert row['status'] == 'confirmed'
        assert json.loads(row['payload'])['input_contract'] == LINK_INPUT_CONTRACT
        assert 'legacy_validation_failure' in json.loads(row['precondition'])
        assert state.performers == ['77', '5'] and len(links(state)) == 1
        assert publish_person(db, client, report, 'scope', '1', 'person') == '5'
        assert len(links(state)) == 1 and db.connection.execute('SELECT COUNT(*) FROM people').fetchone()[0] == 1


@pytest.mark.parametrize('status,error,contract', [
    ('pending', None, False), ('unknown', 'read timed out', False),
    ('unknown', '500 Server Error for url: {url}/graphql', False),
    ('unknown', '400 Client Error: Bad Request for url: http://another/graphql', False),
    ('unknown', '400 Client Error: Bad Request for url: {url}/graphql', True),
])
def test_other_unresolved_links_are_not_retried(tmp_path, status, error, contract):
    with endpoint() as (client, state), FaceDatabase(tmp_path / 'faces.sqlite3') as db:
        person(db)
        payload = {'performer': '5'} | ({'input_contract': LINK_INPUT_CONTRACT} if contract else {})
        save_operation(db, scope='scope', scene='1', person='person', kind='link', target='5',
                       payload=payload, precondition={}, status=status,
                       error=error.format(url=client.url) if error else None)
        with pytest.raises(PendingRecognition, match='no blind retry'):
            publish_person(db, client, RunReport(write=lambda _: None), 'scope', '1', 'person')
        assert not links(state) and state.performers == ['77']


@pytest.mark.parametrize('rejection,outcome', [('validation', 'failed'), ('unclassified', 'unknown')])
def test_400_details_are_bounded_redacted_and_do_not_authorize_uncertain_retries(tmp_path, rejection, outcome):
    with endpoint() as (client, state), FaceDatabase(tmp_path / 'faces.sqlite3') as db:
        person(db)
        state.rejection, state.message = rejection, ('private-key ' * 200) + 'validation error'
        report = RunReport(write=lambda _: None)
        with pytest.raises(requests.HTTPError) as caught:
            publish_person(db, client, report, 'scope', '1', 'person')
        message = str(caught.value)
        assert 'GraphQL:' in message and 'private-key' not in message and '[redacted]' in message
        assert len(message) < 2000 and len(links(state)) == 1
        row = db.connection.execute("SELECT * FROM publications WHERE kind='link'").fetchone()
        assert row['status'] == outcome and 'private-key' not in row['error']
        state.rejection = None
        if outcome == 'unknown':
            with pytest.raises(PendingRecognition):
                publish_person(db, client, report, 'scope', '1', 'person')
            assert len(links(state)) == 1
        else:
            assert publish_person(db, client, report, 'scope', '1', 'person') == '5'
            assert len(links(state)) == 2
