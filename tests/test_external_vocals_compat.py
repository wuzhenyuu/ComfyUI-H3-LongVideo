"""Optional MV vocal input stays compatible without models or GPU work."""
import copy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch

import test_plugin as support


nodes, core = support.nodes, support.core
preparation = importlib.import_module('h3lv_test.audio_preparation')


def audio(wave, rate):
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 1:
        wave = wave[:, None]
    return {'waveform': torch.from_numpy(wave.T.copy()).unsqueeze(0), 'sample_rate': rate}


class ExternalVocalsCompatibilityTests(unittest.TestCase):
    def test_public_optional_input_and_unchanged_output_and_hidden_contract(self):
        schema = nodes.Unified.INPUT_TYPES()
        self.assertEqual(schema['optional']['vocals'][0], 'AUDIO')
        self.assertIn('优先', schema['optional']['vocals'][1]['tooltip'])
        self.assertEqual(schema['hidden'], {'prompt': 'PROMPT', 'unique_id': 'UNIQUE_ID'})
        self.assertEqual(schema['required']['mode'][0], ['singing'])
        self.assertEqual(len(nodes.Unified.RETURN_TYPES), 16)
        self.assertEqual(nodes.Unified.RETURN_NAMES[1], 'vocals_padded')

    def test_unconnected_input_preserves_processing_and_connected_input_is_forwarded(self):
        original = object()
        external = object()
        for vocals in (None, external):
            for mode in preparation.MODES:
                with self.subTest(vocals=vocals, mode=mode), patch.object(
                        nodes.Analyze, 'analyze', return_value={'ui': {}, 'result': ('id', 1)}) as analyze:
                    result = nodes.Unified().process(original, 'singing', 15, 11, '', '',
                                                     vocals=vocals, audio_processing=mode)
                self.assertIs(analyze.call_args.kwargs['vocals'], vocals)
                self.assertEqual(analyze.call_args.kwargs['audio_processing'], mode)
                self.assertEqual(len(result['result']), 16)
                self.assertIs(result['result'][0], original)

    def analyze(self, root, original, external, mode=preparation.EXTRACT, asr=None):
        rules = {'config': support.director_rules.default_config(), 'revision': 'test'}
        samples = original['waveform'].shape[-1]
        asr = asr or Mock(return_value={'segments': [], 'words': []})
        with patch.object(nodes, 'data_root', return_value=root), \
                patch.object(nodes, 'rules_path', return_value=root/'rules'), \
                patch.object(nodes.director_rules, 'read_rules', return_value=rules), \
                patch.object(nodes, 'run_asr', asr), \
                patch.object(nodes, 'segmentation', return_value=([
                    {'start_sample': 0, 'end_sample': samples, 'energy_db': -30, 'text': ''}],
                    {'sections': [], 'rhythm': {}})), \
                patch.object(nodes, 'separate', side_effect=AssertionError('external vocals must skip separation')):
            result = nodes.Unified().process(original, 'singing', 15, 11, '', '',
                                             vocals=external, audio_processing=mode)
        return core.read_plan(root, result['ui']['h3lv_project'][0])

    def test_analysis_uses_external_alignment_and_keeps_full_original_audio(self):
        for mode in preparation.MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                mix = np.column_stack((np.full(48000, .3, np.float32), np.full(48000, -.1, np.float32)))
                external = audio(np.full(44101, .15, np.float32), 44100)
                plan = self.analyze(root, audio(mix, 48000), external, mode)
                directory = core.project_path(root, plan['id'])
                saved_mix, rate = sf.read(directory/'audio/source.wav', dtype='float32', always_2d=True)
                saved_vocals, vocal_rate = sf.read(directory/'audio/vocals.wav', dtype='float32', always_2d=True)
                np.testing.assert_array_equal(saved_mix, mix)
                self.assertEqual((rate, vocal_rate, saved_vocals.shape), (48000, 48000, (48000, 1)))
                expected = preparation.align_reference(*nodes.audio_array(external), 48000, len(mix))
                np.testing.assert_array_equal(saved_vocals, expected)
                self.assertEqual(plan['audio_processing'], 'external')
                self.assertTrue(plan['external_vocals'])
                preparation.verify_reference(directory, plan['audio_preparation'])

    def test_trimmed_external_audio_is_rejected_before_asr(self):
        asr = Mock(return_value={'segments': [], 'words': []})
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, '时长'):
                self.analyze(Path(temporary), audio(np.zeros(1000), 100), audio(np.zeros(900), 100), asr=asr)
            asr.assert_not_called()

    def load(self, root, plan, external=None, mode=preparation.EXTRACT):
        loaded = (None, None, 24, 'prefix', {}, *((None,)*6), '', 24., *((None,)*3))
        with patch.object(nodes, 'data_root', return_value=root), \
                patch.object(nodes, 'read_plan', return_value=copy.deepcopy(plan)), \
                patch.object(nodes.LoadSegment, 'load', return_value=loaded) as load, \
                patch.object(nodes.Analyze, 'analyze', side_effect=AssertionError('must retain approved project')):
            result = nodes.Unified().process({}, 'singing', 15, 11, '', '', project_id=plan['id'],
                                             vocals=external, audio_processing=mode)
        load.assert_called_once_with(plan['id'], 0)
        return result

    def test_approved_external_project_ignores_overridden_widget_but_detects_source_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = audio(np.full(1000, .1, np.float32), 100)
            plan = self.analyze(root, audio(np.zeros(1000), 100), external)
            plan['approved'] = True
            for mode in preparation.MODES:
                self.assertEqual(len(self.load(root, plan, external, mode)), 16)
            for changed in (None, audio(np.full(1000, .2, np.float32), 100)):
                with self.subTest(changed=changed is None), self.assertRaisesRegex(ValueError, '改变|变化'):
                    self.load(root, plan, changed)

    def test_saved_external_mode_without_signature_compares_cached_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = audio(np.full(1000, .1, np.float32), 100)
            plan = self.analyze(root, audio(np.zeros(1000), 100), external)
            plan['approved'] = True
            plan.pop('external_vocals')
            self.load(root, plan, external)
            with self.assertRaisesRegex(ValueError, '改变|变化'):
                self.load(root, plan, audio(np.zeros(1000), 100))

    def test_legacy_external_project_can_reuse_identical_cached_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = audio(np.full(1000, .1, np.float32), 100)
            plan = self.analyze(root, audio(np.zeros(1000), 100), external)
            plan['approved'] = True
            for field in ('audio_processing', 'external_vocals', 'audio_preparation'):
                plan.pop(field)
            self.load(root, plan, external)
            with self.assertRaisesRegex(ValueError, '改变|变化'):
                self.load(root, plan, audio(np.zeros(1000), 100))

    def test_approved_internal_project_rejects_external_or_processing_switch(self):
        plan = support.sample_plan()
        plan.update(approved=True, audio_processing=preparation.EXTRACT)
        self.load(Path('fixture'), plan)
        for changes in ({'external': audio(np.zeros(1000), 100)}, {'mode': preparation.DIRECT}):
            with self.subTest(change=next(iter(changes))), self.assertRaisesRegex(ValueError, '改变|变化'):
                self.load(Path('fixture'), plan, **changes)


if __name__ == '__main__':
    unittest.main()
