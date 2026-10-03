"""Paginated Stash tag catalog used by cleanup discovery."""
from tqdm import tqdm

PAGE_SIZE = 1000


class CleanupClient:
    def __init__(self, stash):
        self.stash = stash

    def catalog(self):
        catalog = {}
        with tqdm(desc='Read cleanup tags', unit='tag', dynamic_ncols=True) as bar:
            page = 1
            while True:
                count = 'count' if page == 1 else ''
                response = self.stash.gql('''query StashAICleanupTags($filter: FindFilterType!) {
                    findTags(filter: $filter) { ''' + count + ''' tags { id name } }
                }''', {'filter': self.filter(page)})['findTags']
                for tag in response['tags']:
                    if not isinstance(tag.get('name'), str) or not tag.get('id'):
                        raise ValueError('Incomplete tag catalog; cleanup stopped.')
                    catalog[str(tag['id'])] = tag['name']
                bar.total = max(response.get('count', bar.total or 0), bar.n + len(response['tags']))
                bar.update(len(response['tags']))
                if len(response['tags']) < PAGE_SIZE:
                    return catalog
                page += 1

    @staticmethod
    def filter(page=1, size=None):
        return dict(page=page, per_page=PAGE_SIZE if size is None else size, sort='id', direction='ASC')

