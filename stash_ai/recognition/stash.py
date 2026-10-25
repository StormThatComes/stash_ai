"""Face-only Stash reads and validated performer/link/merge mutations."""
from stash_ai.stash_api.transport import StashTransport

PERFORMER_FIELDS = 'id name details image_path'
SCENE_FIELDS = 'id title paths { vtt sprite stream } performers { id }'


def checked_id(value, expected=None):
    if (not isinstance(value, dict) or not isinstance(value.get('id'), str) or
            not value['id'].isascii() or not value['id'].isdecimal() or int(value['id']) <= 0 or
            value['id'] != str(int(value['id'])) or
            expected is not None and value['id'] != str(expected)):
        raise RuntimeError('Stash performer/scene acknowledgement ID was not confirmed.')
    return value


def checked_token(database, person, token):
    expected = f'stash-ai-recognition:{database.installation}:{person}'
    if token != expected:
        raise ValueError('Recognition ownership token does not match installation and person identity.')
    return token


def owns(performer, token):
    return isinstance(performer, dict) and token in (performer.get('details') or '').splitlines()


class RecognitionStash(StashTransport):
    def __init__(self, *, url, api_key, timeout=120):
        super().__init__(False, url=url, api_key=api_key, request_timeout=(5., timeout), http_retries=2)
        try:
            from stash_ai.stash_api.client import StashClient
            StashClient._require_supported_version(self)
        except BaseException:
            from stash_ai.runtime.cleanup import cleanup
            cleanup([('close failed recognition Stash client', self.close)])
            raise

    def new_media_reader(self):
        reader = RecognitionReader(True, url=self.url, api_key=self.api_key,
                                   request_timeout=self.request_timeout, http_retries=self.http_retries)
        reader.report = self.report
        return reader

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
        return checked_id(self.lookup_performer(identifier), identifier)

    def lookup_performer(self, identifier):
        value = self.gql('query StashAIRecognitionPerformer($id:ID!) { findPerformer(id:$id) { '
                         + PERFORMER_FIELDS + ' } }', {'id': identifier})['findPerformer']
        return None if value is None else checked_id(value, identifier)

    def _read_batch(self, identifiers, *, scenes=False):
        values=list(dict.fromkeys(str(value) for value in identifiers))
        for identifier in values:
            checked_id({'id':identifier})
        result={}
        for offset in range(0,len(values),32):
            batch=values[offset:offset+32]
            declarations=','.join(f'$id{i}:ID!' for i in range(len(batch)))
            field,fields=('findScene',SCENE_FIELDS) if scenes else ('findPerformer',PERFORMER_FIELDS)
            selections=' '.join(f'n{i}:{field}(id:$id{i}) {{ {fields} }}' for i in range(len(batch)))
            operation='StashAIRecognitionScenesBatch' if scenes else 'StashAIRecognitionPerformersBatch'
            try:
                payload=self.gql(f'query {operation}({declarations}) {{ {selections} }}',
                                 {f'id{i}':value for i,value in enumerate(batch)})
                if not isinstance(payload,dict) or set(payload)!={f'n{i}' for i in range(len(batch))}:
                    raise RuntimeError('Stash batch read did not return the expected aliases.')
            except Exception:
                # A field error can reject a complete GraphQL envelope. Isolate
                # entries through the existing proven-read/retry surface.
                for value in batch:
                    try:
                        result[value]=self.scene(value) if scenes else self.lookup_performer(value)
                    except Exception as error:
                        result[value]=error
                continue
            for index,value in enumerate(batch):
                try:
                    row=payload[f'n{index}']
                    result[value]=None if row is None else checked_id(row,value)
                except Exception as error:
                    result[value]=error
        return result

    def scenes(self, identifiers):
        return self._read_batch(identifiers,scenes=True)

    def performers(self, identifiers):
        return self._read_batch(identifiers)

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
                        {'input': {'ids': [scene], 'performer_ids': {'ids': [performer], 'mode': 'ADD'}}})['bulkSceneUpdate']
        if not isinstance(rows, list) or len(rows) != 1:
            raise RuntimeError('Stash additive performer association was not acknowledged.')
        checked_id(rows[0], scene)
        if performer not in {checked_id(row)['id'] for row in rows[0].get('performers', [])}:
            raise RuntimeError('Stash did not confirm the requested performer association.')
        return rows[0]

    def add_scene_people(self,scene,performers):
        checked_id({'id':scene})
        wanted=list(dict.fromkeys(performers))
        for identifier in wanted:checked_id({'id':identifier})
        if not wanted:return self.scene(scene)
        rows=self.gql('mutation StashAIRecognitionLinks($input:BulkSceneUpdateInput!) {'
            'bulkSceneUpdate(input:$input) { id performers { id } }}',
            {'input':{'ids':[scene],'performer_ids':{'ids':wanted,'mode':'ADD'}}})['bulkSceneUpdate']
        if not isinstance(rows,list) or len(rows)!=1:
            raise RuntimeError('Stash additive associations were not acknowledged.')
        checked_id(rows[0],scene)
        if not set(wanted)<={checked_id(row)['id'] for row in rows[0].get('performers',[])}:
            raise RuntimeError('Stash did not confirm every requested association.')
        return rows[0]

    def update_person(self, performer, **fields):
        if set(fields) - {'name', 'image'}:
            raise ValueError('Recognition may update only selected performer name/image fields.')
        value = self.gql('mutation StashAIRecognitionMetadata($input:PerformerUpdateInput!) {'
                         'performerUpdate(input:$input) { ' + PERFORMER_FIELDS + ' }}',
                         {'input': {'id': performer, **fields}})['performerUpdate']
        return checked_id(value, performer)

    def remove_scene_person(self, scene, performer):
        rows = self.gql('mutation StashAIRecognitionUnlink($input:BulkSceneUpdateInput!) {'
                        'bulkSceneUpdate(input:$input) { id performers { id } }}',
                        {'input': {'ids': [scene], 'performer_ids': {'ids': [performer], 'mode': 'REMOVE'}}})['bulkSceneUpdate']
        if not isinstance(rows, list) or len(rows) != 1:
            raise RuntimeError('Stash performer disassociation was not acknowledged.')
        checked_id(rows[0], scene)
        if not isinstance(rows[0].get('performers'), list):
            raise RuntimeError('Stash performer disassociation lacks an association acknowledgement.')
        if performer in {checked_id(row)['id'] for row in rows[0]['performers']}:
            raise RuntimeError('Stash did not confirm the requested performer disassociation.')
        return rows[0]

    def merge_person(self, source, destination, details):
        checked_id({'id': source})
        checked_id({'id': destination})
        if source == destination:
            raise ValueError('Recognition cannot merge a performer into itself.')
        value = self.gql('mutation StashAIRecognitionMerge($input:PerformerMergeInput!) {'
                         'performerMerge(input:$input) { ' + PERFORMER_FIELDS + ' }}',
                         {'input': {'source': [source], 'destination': destination,
                                    'values': {'id': destination, 'details': details}}})['performerMerge']
        return checked_id(value, destination)

    def recognition_performers(self, *, maximum=100000):
        """Enumerate token-bearing Stash profiles without treating names as ownership."""
        page, total, seen = 1, 0, set()
        while True:
            payload = self.gql(
                'query StashAIRecognitionOwnedProfiles($filter:FindFilterType!,$needle:String!) {'
                'findPerformers(performer_filter:{details:{value:$needle,modifier:INCLUDES}},'
                'filter:$filter) { performers { ' + PERFORMER_FIELDS + ' } }}',
                {'needle': 'stash-ai-recognition:',
                 'filter': {'page': page, 'per_page': 100, 'sort': 'id', 'direction': 'ASC'}})
            rows = payload['findPerformers']['performers']
            if not isinstance(rows, list) or len(rows) > 100:
                raise RuntimeError('Invalid recognition performer inventory page.')
            for row in rows:
                checked_id(row)
                if row['id'] in seen:
                    raise RuntimeError('Recognition performer inventory repeated a performer.')
                seen.add(row['id'])
                total += 1
                if total > maximum:
                    raise ValueError('Recognition performer inventory exceeds the supported limit.')
                yield row
            if len(rows) < 100:
                return
            page += 1

    def delete_person(self, performer):
        checked_id({'id': str(performer)})
        result = self.gql('mutation StashAIRecognitionDelete($input:PerformerDestroyInput!) {'
                          'performerDestroy(input:$input) }', {'input': {'id': str(performer)}})
        if result.get('performerDestroy') is not True:
            raise RuntimeError('Stash did not acknowledge recognition performer deletion.')
        return True


class RecognitionReader(StashTransport):
    """Isolated read-only HTTP sessions with the same validated scene query."""
    scene = RecognitionStash.scene
    performer = RecognitionStash.performer
    lookup_performer = RecognitionStash.lookup_performer
    _read_batch = RecognitionStash._read_batch
    scenes = RecognitionStash.scenes
    performers = RecognitionStash.performers
