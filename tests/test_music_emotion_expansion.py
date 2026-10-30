"""Music analysis stays a foundation input, independent of automatic briefs."""
import copy
import importlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import test_materials as fixtures
from test_expansion_fixtures import chinese_prompt, confirmed_packet, REMOTE_TEXT

expansion = fixtures.expansion
cli = importlib.import_module('h3lv_test.codex_cli')
ANALYSIS = {'labels': ['平静', '克制'], 'intensity': 2.5,
            **{name: '音乐表达估计：'+name for name in
               ('trend', 'summary', 'melody', 'rhythm', 'timbre', 'vocal', 'lyrics', 'uncertainty')}}
PROFILE = {'base_url': 'https://fixture.test/v1'}


class MusicEmotionExpansionTests(unittest.TestCase):
    setUp = fixtures.MaterialTests.setUp

    def packet(self):
        row = self.plan['segments'][0]
        row.update(shot_request='@图1 是人物，@图2 是场景。人物向左走，镜头跟随。',
                   music_emotion={'status': 'ready', 'analysis': copy.deepcopy(ANALYSIS),
                                  'raw': 'private raw output', 'source_path': 'private.wav'})
        return fixtures.materials.packet(self.plan, row, self.directory)

    def test_ready_analysis_survives_explicit_intent_and_stale_analysis_is_excluded(self):
        packet = self.packet()
        self.assertEqual(packet['brief'], '')
        self.assertEqual(packet['brief_source'], 'none')
        self.assertEqual(packet['music_emotion'], ANALYSIS)
        self.assertEqual(len(packet['music_emotion']), 10)
        self.assertNotIn('raw', json.dumps(packet, ensure_ascii=False))
        self.plan['segments'][0]['music_emotion']['status'] = 'stale'
        stale = fixtures.materials.packet(self.plan, self.plan['segments'][0], self.directory)
        self.assertNotIn('music_emotion', stale)

    def test_api_and_cli_receive_analysis_without_audio_permission_or_dialogue(self):
        for provider in ('api', 'codex_cli'):
            packet = self.packet()
            profile = PROFILE if provider == 'api' else {'cli_identity': {'backend': 'fixture'}}
            with self.subTest(provider=provider), \
                 patch.object(expansion, 'call', return_value={'choices':[{'message':{'content':chinese_prompt()}}]}) as api, \
                 patch.object(cli, 'generate', return_value=chinese_prompt()) as codex:
                result = expansion.expand(packet, 'text', 'fixture', 'style', provider=provider,
                    request_profile=profile, cancelled=lambda: False)
            if provider == 'api':
                payload = api.call_args.args[1]
                context = json.loads(payload['messages'][1]['content'][0]['text'])
                guard = payload['messages'][0]['content']
            else:
                guard = codex.call_args.args[0]
                context = json.loads(guard.split('分段素材包：\n')[1])
            self.assertEqual(context['music_emotion'], ANALYSIS)
            self.assertEqual(context['brief'], '')
            self.assertIn('向左走', context['material_note'])
            self.assertIs(context['audio_reference_available'], False)
            self.assertNotIn('<Audio', result)
            self.assertIn('music-expression foundation', guard)
            self.assertIn('never an audio input', guard)

    def test_each_analysis_field_invalidates_api_and_cli_cache_identity(self):
        packet = self.packet()
        for provider, profile in (('api', PROFILE), ('codex_cli', {'cli_identity': {'backend': 'fixture'}})):
            def key(value):
                return expansion.cache_key(value, 'text', 'fixture', 'style', 0, profile=profile, provider=provider)
            original = key(packet)
            for field in ANALYSIS:
                changed = copy.deepcopy(packet)
                changed['music_emotion'][field] = (['期待'] if field == 'labels' else 4 if field == 'intensity' else '变化后的估计')
                with self.subTest(provider=provider, field=field):
                    self.assertNotEqual(original, key(changed))
            stripped = {k:v for k,v in packet.items() if k != 'music_emotion'}
            self.assertNotEqual(original, key(stripped))

    def test_unverified_performance_never_gains_audio_from_analysis_or_legacy_cache(self):
        packet = self.packet()
        no_audio = fixtures.MULTI_TEXT.replace('<Audio 1> is singing.', '').replace('follows <Audio 1>', 'shows the performer')
        self.assertNotIn('<Audio', expansion.validate_result(no_audio, packet))
        for updates in ({}, {'audio_reference_available': True, 'audio_reference_slot': True},
                        {'audio_reference_available': True, 'audio_reference_slot': 2}):
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, 'Audio 1'):
                expansion.validate_result(fixtures.MULTI_TEXT, {**packet, **updates})

    def test_invalid_saved_text_and_new_english_output_are_not_success_cache(self):
        packet = self.packet()
        with patch.object(expansion, 'public_settings', return_value=PROFILE):
            key = expansion.cache_key(packet, 'text', 'fixture', '', 0)
            packet.update(expanded_key=key, expanded_prompt=fixtures.MULTI_TEXT)
            self.assertEqual(expansion.cache_source(packet, key, 'text'), 'api')
            english = fixtures.MULTI_TEXT.replace('<Audio 1> is singing.', '').replace('follows <Audio 1>', 'shows the performer')
            for _ in range(2):
                with patch.object(expansion, 'call', return_value={'choices':[{'message':{'content':english}}]}) as request:
                    with self.assertRaises(ValueError):
                        expansion.expand(packet, 'text', 'fixture', '')
                    request.assert_called_once()
            self.assertEqual(expansion.cache_source(packet, key, 'text'), 'api')
        success = [p for p in Path(packet['cache_dir']).glob('*.json') if not p.name.endswith('.invalid.json')]
        self.assertEqual(success, [])

    def test_confirmed_legacy_audio_and_manual_request_remain_usable(self):
        packet = self.packet()
        confirmed = {**packet, 'audio_reference_available': True, 'audio_reference_slot': 1}
        self.assertEqual(expansion.validate_result(fixtures.MULTI_TEXT, confirmed), fixtures.MULTI_TEXT)
        with patch.object(expansion, 'public_settings', return_value=PROFILE), patch.object(expansion, 'call') as request:
            self.assertEqual(expansion.expand(packet, 'manual', 'fixture', ''), packet['material_note'])
        request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
