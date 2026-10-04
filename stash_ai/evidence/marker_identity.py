"""Reserved marker names and classification, independent of evidence algorithms."""
OCCURRENCE_TAG = 'ai:marker:occurrence'


def is_occurrence(marker):
    return bool(marker.get('contributors')) and all(
        item.get('derived') == 'occurrence' for item in marker['contributors'])
