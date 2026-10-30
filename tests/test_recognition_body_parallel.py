from functools import partial
import pytest
from stash_ai.recognition.compute import compute_scope
from tests.test_recognition_compute import configured
from tests import test_recognition_body_scaling as oracle
from tests import test_recognition_body_global as guards


@pytest.mark.parametrize('case', [
    partial(oracle.test_sparse_planner_matches_exhaustive_decisions_and_pair_proofs, seed=5),
    oracle.test_duplicate_core_checks_share_scores_but_keep_proof_multiplicity,
    oracle.test_requested_subthreshold_pairs_and_all_goal_rivals_are_exact,
    oracle.test_cross_tile_candidates_rivals_and_small_working_budget_match_oracle,
    oracle.test_original_core_veto_survives_nomination_and_score_reuse,
    oracle.test_near_threshold_centroid_rounding_does_not_change_acceptance,
    guards.test_weak_rival_still_blocks_close_pair,
    guards.test_contradictory_faces_veto_matching_bodies,
])
def test_parallel_body_proofs_against_exhaustive_oracle(tmp_path, monkeypatch, case):
    monkeypatch.setattr('stash_ai.recognition.compute.available_cpus', lambda: 16)
    with compute_scope(configured(rebuild_workers=4)) as pool:
        case(tmp_path)
        assert pool.submitted > 0
        assert not pool.pending
        assert pool.peak_bytes <= pool.limit


@pytest.mark.parametrize('limit', [1, 40])
def test_parallel_witness_budget_fails_without_mutation(tmp_path, monkeypatch, limit):
    monkeypatch.setattr('stash_ai.recognition.compute.available_cpus', lambda: 16)
    config, state = oracle.fixture(tmp_path, [[1, 0]]*5)
    try:
        before = config.database.read_bytes()
        config.data['execution']['rebuild_workers'] = 4
        config.data['sapiensid']['execution']['max_witness_comparisons'] = limit
        with compute_scope(config), pytest.raises(ValueError, match='witness comparisons'):
            oracle.report(state.connection, config, 'scope', 'ns')
        assert config.database.read_bytes() == before
    finally:
        state.__exit__(None, None, None)
