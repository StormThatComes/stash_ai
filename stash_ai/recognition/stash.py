"""Face-only Stash reads and validated additive performer mutations."""
from stash_ai.stash_api.transport import StashTransport

PERFORMER_FIELDS = 'id name details image_path'
SCENE_FIELDS = 'id title paths { vtt sprite stream } performers { id }'


def checked_id(value, expected=None):
    if (not isinstance(value, dict) or not isinstance(value.get('id'), str) or
            not value['id'].isascii() or not value['id'].isdecimal() or int(value['id']) <= 0 or
            expected is not None and value['id'] != str(expected)):
        raise RuntimeError('Stash performer/scene acknowledgement ID was not confirmed.')
    return value


def owns(performer, token):
    return isinstance(performer, dict) and token in (performer.get('details') or '').splitlines()


class RecognitionStash(StashTransport):
    def __init__(self, *, url, api_key, timeout=120):
        super().__init__(False, url=url, api_key=api_key, request_timeout=(5., timeout), http_retries=2)
        try:
            from stash_ai.stash_api.client import StashClient
            StashClient._require_supported_version(self)
        except BaseException:
            self.close()
            raise

    def scene(self, identifier):
        value = self.gql('query StashAIRecognitionScene($id:ID!) { findScene(id:$id) { '
                         + SCENE_FIELDS + ' } }', {'id': identifier})['findScene']
        return checked_id(value, identifier)

    def scene_ids(self, *, maximum=100000):
        page, count = 1, 0
        while True:
            payload = self.gql('query StashAIRecognitionSceneIDs($filter:FindFilterType!) {'
                               'findScenes(filter:$filter) { scenes { id } }}',
                               {'filter': {'page': page, 'per_page': 1000, 'sort': 'id', 'direction': 'ASC'}})
            rows = payload['findScenes']['scenes']
            for row in rows:
                checked_id(row)
                count += 1
                if count > maximum:
                    raise ValueError('Recognition scene selection exceeds the supported limit.')
                yield row['id']
            if len(rows) < 1000:
                return
            page += 1

    def performer(self, identifier):
        value = self.gql('query StashAIRecognitionPerformer($id:ID!) { findPerformer(id:$id) { '
                         + PERFORMER_FIELDS + ' } }', {'id': identifier})['findPerformer']
        return checked_id(value, identifier)

    def find_token(self, token):
        payload = self.gql('query StashAIRecognitionOwnership($token:String!) {'
                           'findPerformers(performer_filter:{details:{value:$token,modifier:INCLUDES}},'
                           'filter:{per_page:100,page:1}) { performers { ' + PERFORMER_FIELDS + ' } }}', {'token': token})
        rows = payload['findPerformers']['performers']
        if len(rows) >= 100:
            raise RuntimeError('Recognition ownership lookup is ambiguous/over budget.')
        return [checked_id(row) for row in rows if owns(row, token)]

    def create_person(self, name, token):
        value = self.gql('mutation StashAIRecognitionCreate($input:PerformerCreateInput!) {'
                         'performerCreate(input:$input) { ' + PERFORMER_FIELDS + ' }}',
                         {'input': {'name': name, 'details': token}})['performerCreate']
        checked_id(value)
        if not owns(value, token):
            raise RuntimeError('Created performer ownership was not acknowledged.')
        return value

    def add_scene_person(self, scene, performer):
        rows = self.gql('mutation StashAIRecognitionLink($input:BulkSceneUpdateInput!) {'
                        'bulkSceneUpdate(input:$input) { id performers { id } }}',
                        {'input': {'ids': [scene], 'performer_ids': {'values': [performer], 'mode': 'ADD'}}})['bulkSceneUpdate']
        if not isinstance(rows, list) or len(rows) != 1:
            raise RuntimeError('Stash additive performer association was not acknowledged.')
        checked_id(rows[0], scene)
        if performer not in {checked_id(row)['id'] for row in rows[0].get('performers', [])}:
            raise RuntimeError('Stash did not confirm the requested performer association.')
        return rows[0]

    def update_person(self, performer, **fields):
        if set(fields) - {'name', 'image'}:
            raise ValueError('Recognition may update only selected performer name/image fields.')
        value = self.gql('mutation StashAIRecognitionMetadata($input:PerformerUpdateInput!) {'
                         'performerUpdate(input:$input) { ' + PERFORMER_FIELDS + ' }}',
                         {'input': {'id': performer, **fields}})['performerUpdate']
        return checked_id(value, performer)
