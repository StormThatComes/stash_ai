"""Compatibility boundary for extracted ONNX tagger runtime."""
from stash_ai import onnx_tagger
from stash_ai import taggers


def test_taggers_reexports_onnx_runtime():
    assert taggers.OnnxTagger is onnx_tagger.OnnxTagger
    assert taggers.prepare_oppai is onnx_tagger.prepare_oppai
    assert taggers.prepare_wd is onnx_tagger.prepare_wd
    assert taggers.prepare_dtq is onnx_tagger.prepare_dtq
