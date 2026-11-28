import sqlite3

from stash_ai.recognition.prior_membership import PriorMembership


def test_previous_membership_streams_and_has_framed_digest():
    db = sqlite3.connect(':memory:')
    first = PriorMembership(db,'first_ids',iter(['a','bc']))
    other = PriorMembership(db,'other_ids',iter(['ab','c']))
    assert first and 'bc' in first and 'ab' not in first
    assert first.digest != other.digest
    assert not PriorMembership(db,'empty_ids',iter(()))
    db.close()
