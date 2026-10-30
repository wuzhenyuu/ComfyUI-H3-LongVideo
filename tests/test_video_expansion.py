"""Visual video-reference expansion: no model service or media decoder is run."""
import base64
import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import test_materials as fixtures

expansion = fixtures.expansion
rules = expansion.h3_prompt_rules
cli = importlib.import_module('h3lv_test.codex_cli')
jobs = importlib.import_module('h3lv_test.expansion_jobs')
PROFILE = {'base_url': 'https://fixture.test/v1', 'api_key': 'private-fixture'}
CLI_PROFILE = {'cli_identity': {'backend': 'local-fixture', 'model': '', 'reasoning': 'default'}}


def video_prompt(count=1):
    text = fixtures.TEXT
    definitions = ''.join(f'\n<Video {i}> 是缓慢前推运镜与镜头渐缓节奏的视觉参考，仅用于本段可用帧。' for i in range(1, count+1))
    retention = ''.join(f'\n<Video {i}>：weak_reference - 仅参考可用视觉样本的运镜方向与节奏。' for i in range(1, count+1))
    text = text.replace('\n\nsummary:', definitions+'\n\nsummary:')
    text = text.replace('\n\ndetailed_description:', retention+'\n\ndetailed_description:')
    return text.replace(fixtures.SHOT_BODY, '<Video 1> 的视觉运镜参考从本段开场生效，'+fixtures.SHOT_BODY)


class VideoExpansionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.pictures, self.sheets = [], []
        for index, color in enumerate(((220, 0, 0), (0, 210, 0), (0, 0, 220), (200, 200, 0))):
            path = self.directory/f'image-{index}.png'
            Image.new('RGB', (32, 24), color).save(path)
            (self.pictures if index < 2 else self.sheets).append(str(path))
        self.packet = dict(paths=self.pictures, hashes=['picture-one', 'picture-two'],
            material_note='@图1是人物，@图2是环境', visual_type='atmosphere', mode='singing',
            brief='只参考声明的慢速前推，人物嘴唇闭合。', duration=10,
            generation_frames=242, generation_seconds=242/24, audio_role='instrumental',
            audio_section='interlude', audio_role_reason='fixture', cache_dir=str(self.directory/'cache'),
            video_refs=['video-one', 'video-two'], video_paths=['one.mp4', 'two.mp4'],
            video_hashes=['video-hash-one', 'video-hash-two'], video_durations=[3.0, 4.0],
            video_note='@视频1参考缓慢前推，@视频2参考动作节奏', video_reference_source='default')

    def one_video(self):
        packet = copy.deepcopy(self.packet)
        for field in ('video_refs', 'video_paths', 'video_hashes', 'video_durations'):
            packet[field] = packet[field][:1]
        packet['video_note'] = '@视频1参考慢速前推'
        return packet

    def expand(self, packet=None, mode='vision', provider='api', answer=None):
        return expansion.expand(self.packet if packet is None else packet, mode, 'fixture', '规则',
            provider=provider, request_profile=CLI_PROFILE if provider == 'codex_cli' else PROFILE,
            cancelled=lambda: False)

    def test_real_video_slots_allow_motion_and_require_every_selected_source(self):
        text = video_prompt(2)
        self.assertEqual(expansion.validate_prompt(text, 2, self.packet), text)
        for invalid in (fixtures.TEXT, video_prompt(), text.replace('<Video 2>', '<Video 4>'),
                        text.replace('<Video 1>', '<Video 0>')):
            with self.subTest(invalid=invalid[:100]), self.assertRaises(ValueError):
                expansion.validate_prompt(invalid, 2, self.packet)
        with self.assertRaisesRegex(ValueError, '虚构 Video'):
            expansion.validate_prompt(video_prompt(), 2)

    def test_action_provenance_uses_subject_and_camera_source_requires_retention(self):
        packet = self.one_video()
        action = fixtures.TEXT.replace('场景、光影与构图参考。', '人物身份与服装参考，手臂动作借用 <Video 1>。')
        expansion.validate_prompt(action, 2, packet)
        with self.assertRaisesRegex(ValueError, '动作'):
            expansion.validate_prompt(action.replace('手臂动作', '人物身份'), 2, packet)
        with self.assertRaisesRegex(ValueError, 'retention'):
            expansion.validate_prompt(video_prompt().replace('<Video 1>：weak_reference - 仅参考可用视觉样本的运镜方向与节奏。', ''), 2, packet)
        with self.assertRaisesRegex(ValueError, '整段运镜'):
            expansion.validate_prompt(video_prompt().replace('缓慢前推运镜与镜头渐缓节奏的视觉参考', '源片中的人物与服装参考'), 2, packet)

    def test_video_metadata_and_ui_mentions_fail_before_provider_or_contact_sheet(self):
        invalid_packets = []
        for field in ('video_hashes', 'video_durations'):
            invalid = self.one_video()
            invalid.pop(field)
            invalid_packets.append(invalid)
        for values in ([1.9], [15.1], [True], [float('nan')], [float('inf')]):
            invalid_packets.append({**self.one_video(), 'video_durations': values})
        invalid_packets.extend([
            {**self.packet, 'video_durations': [8.0, 8.0]},
            {**self.packet, 'video_refs': ['missing-second']},
            {**self.packet, 'video_hashes': ['', 'two']},
            {**self.one_video(), 'video_note': '@视频0是运镜'},
            {**self.one_video(), 'video_note': '@视频4是运镜'},
            {**self.one_video(), 'video_note': '<Video 2> 是运镜'},
            {**self.packet, 'video_paths': 'one.mp4'},
            {**self.packet, 'video_paths': ['a', 'b', 'c', 'd']},
        ])
        for packet in invalid_packets:
            with self.subTest(packet=packet), patch.object(expansion, 'call') as api, \
                 patch.object(cli, 'generate') as generate, patch.object(expansion, 'video_contact_sheets') as sheets:
                for provider in ('api', 'codex_cli'):
                    with self.assertRaises(ValueError):
                        self.expand(packet, provider=provider)
                api.assert_not_called()
                generate.assert_not_called()
                sheets.assert_not_called()

    def test_video_cannot_enable_audio_or_change_task_to_editing(self):
        packet = self.one_video()
        valid = video_prompt()
        for phrase in ('沿用 <Video 1> 的原声。', '参考 <Video 1> 的人声音色。',
                       '根据 <Video 1> 中的歌词演唱。', '完整保留视频中的声轨。', '使用视频原声作为伴奏。'):
            text = valid.replace(fixtures.SHOT_BODY, phrase+fixtures.SHOT_BODY)
            with self.subTest(phrase=phrase), self.assertRaisesRegex(ValueError, '原声|声轨'):
                expansion.validate_prompt(text, 2, packet)
        text = valid.replace(fixtures.SHOT_BODY, '不参考 <Video 1> 的原声，也不复制视频中的音轨。'+fixtures.SHOT_BODY)
        expansion.validate_prompt(text, 2, packet)
        for phrase in ('<Video 1> 只作视觉参考，无原声。', '<Video 1> 的原声不参与参考。'):
            expansion.validate_prompt(valid.replace(fixtures.SHOT_BODY, phrase+fixtures.SHOT_BODY), 2, packet)
        with self.assertRaisesRegex(ValueError, 'reference generation'):
            expansion.validate_prompt(valid.replace('[reference generation]', '[keyframe completion]'), 2, packet)
        for task in ('video editing', 'video continuation', 'audio reference', 'audio reuse'):
            with self.subTest(task=task), self.assertRaises(ValueError):
                expansion.validate_prompt(valid.replace('[reference generation]', '[reference generation + '+task+']'), 2, packet)
        for tag in ('<Audio 1>', 'fully_copy', 'partially_copy'):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                expansion.validate_prompt(valid.replace(fixtures.SHOT_BODY, tag+' '+fixtures.SHOT_BODY), 2, packet)

    def project_audio_prompt(self):
        text = video_prompt().replace('场景、光影与构图参考。', '项目表演者的身份、服装与环境参考。')
        text = text.replace('\n\nsummary:', '\n<Audio 1> 是 <Subject 1> (S1) 已核实的独立项目演唱人声参考。\n\nsummary:')
        text = text.replace('[reference generation]', '[reference generation + audio reference]')
        text = text.replace('\n\ndetailed_description:', '\n<Audio 1>：reference - 参考项目原有演唱人声，保持已确认的人声关系。\n\ndetailed_description:')
        return text.replace(fixtures.SHOT_BODY, '<Subject 1> (S1) 结合 <Video 1> 的手臂动作与独立项目 <Audio 1> 人声进行自然口型表演。')

    def test_video_preserves_confirmed_independent_project_vocal_audio(self):
        packet = {**self.one_video(), 'visual_type': 'performance',
            'audio_reference_available': True, 'audio_reference_slot': 1}
        text = self.project_audio_prompt()
        self.assertEqual(expansion.validate_prompt(text, 2, packet), text)
        self.assertIn('不移除或改写已经核实的项目人声', expansion.generation_guard(packet))
        with patch.object(expansion, 'call', return_value={'choices': [{'message': {'content': text}}]}) as provider:
            self.assertEqual(self.expand(packet, mode='text'), text)
        provider.assert_called_once()

    def test_video_source_audio_is_rejected_even_with_confirmed_project_slot(self):
        packet = {**self.one_video(), 'visual_type': 'performance',
            'audio_reference_available': True, 'audio_reference_slot': 1}
        text = self.project_audio_prompt()
        for source in ('来自 <Video 1> 的原声', '音色参考自 <Video 1>', '来自 <Video 1> 的同步音轨',
                       '来自 <Video 1> 的背景音乐', '来自 <Video 1> 的音效'):
            invalid = text.replace('已核实的独立项目演唱人声参考', source)
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, '参考视频的原声'):
                expansion.validate_prompt(invalid, 2, packet)

    def test_video_does_not_bypass_project_audio_slot_or_visual_type_gates(self):
        packet = {**self.one_video(), 'visual_type': 'performance',
            'audio_reference_available': True, 'audio_reference_slot': 1}
        for updates in ({'audio_reference_available': False}, {'audio_reference_slot': None},
                        {'audio_reference_slot': 2}, {'audio_reference_slot': True},
                        {'visual_type': 'atmosphere'}, {'visual_type': 'environment'}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                expansion.validate_prompt(self.project_audio_prompt(), 2, {**packet, **updates})

    def test_source_longer_than_segment_cannot_be_promised_in_full(self):
        packet = {**self.one_video(), 'video_durations': [15.0]}
        text = video_prompt().replace(fixtures.SHOT_BODY, '从头到尾完整使用 <Video 1> 的15秒源视频。'+fixtures.SHOT_BODY)
        with self.assertRaisesRegex(ValueError, '超出本段'):
            expansion.validate_prompt(text, 2, packet)
        expansion.validate_prompt(video_prompt(), 2, packet)

    def test_atmosphere_keeps_existing_no_vocal_gate_with_video(self):
        for kind in ('atmosphere', 'environment'):
            packet = {**self.one_video(), 'visual_type': kind}
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, '口型'):
                expansion.validate_prompt(video_prompt().replace(fixtures.SHOT_BODY, '人物开口说话并做口型。'), 2, packet)

    def test_api_vision_sends_pictures_then_one_contact_sheet_per_video(self):
        with patch.object(expansion, 'video_contact_sheets', return_value=self.sheets) as sheets, \
             patch.object(expansion, 'video_sample_timestamps', return_value=[i/24 for i in range(8)]), \
             patch.object(expansion, 'call', return_value={'choices': [{'message': {'content': video_prompt(2)}}]}) as provider:
            self.assertEqual(self.expand(), video_prompt(2))
        sheets.assert_called_once_with(self.packet)
        payload = provider.call_args.args[1]
        content = payload['messages'][1]['content']
        context = json.loads(content[0]['text'])
        self.assertEqual(context['material_note'], '<Picture 1>是人物，<Picture 2>是环境')
        self.assertEqual(context['video_note'], '<Video 1>参考缓慢前推，<Video 2>参考动作节奏')
        self.assertEqual(context['pictures'], [{'number': 1}, {'number': 2}])
        self.assertEqual([video['label'] for video in context['videos']], ['<Video 1>', '<Video 2>'])
        self.assertEqual([video['source_duration_seconds'] for video in context['videos']], [3.0, 4.0])
        self.assertTrue(all(len(video['sample_timestamps_seconds']) == 8 for video in context['videos']))
        colors = []
        for item in content[1:]:
            self.assertEqual(item['type'], 'image_url')
            raw = base64.b64decode(item['image_url']['url'].partition(',')[2])
            with Image.open(io.BytesIO(raw)) as image:
                colors.append(image.getpixel((0, 0)))
        self.assertEqual(len(colors), 4)
        for actual, expected in zip(colors, [(220, 0, 0), (0, 210, 0), (0, 0, 220), (200, 200, 0)]):
            self.assertTrue(all(abs(a-b) <= 3 for a, b in zip(actual, expected)))
        self.assertNotIn('video_url', str(payload))

    def test_cli_vision_preserves_attachment_order_and_text_has_no_attachments(self):
        with patch.object(expansion, 'video_contact_sheets', return_value=self.sheets) as sheets, \
             patch.object(expansion, 'video_sample_timestamps', return_value=[0.0]*8), \
             patch.object(cli, 'generate', return_value=video_prompt(2)) as generate:
            self.expand(provider='codex_cli')
        self.assertEqual(generate.call_args.args[1], self.pictures+self.sheets)
        self.assertIn('只可根据这些抽样帧', generate.call_args.args[0])
        sheets.assert_called_once()
        for provider in ('api', 'codex_cli'):
            packet = {**self.packet, 'cache_dir': str(self.directory/('text-'+provider))}
            with self.subTest(provider=provider), \
                 patch.object(expansion, 'video_contact_sheets', side_effect=AssertionError('Text sampled video')), \
                 patch.object(expansion, 'video_sample_timestamps', side_effect=AssertionError('Text sampled video')), \
                 patch.object(expansion, 'call', return_value={'choices': [{'message': {'content': video_prompt(2)}}]}) as api, \
                 patch.object(cli, 'generate', return_value=video_prompt(2)) as generate:
                self.expand(packet, mode='text', provider=provider)
                if provider == 'api':
                    content = api.call_args.args[1]['messages'][1]['content']
                    self.assertEqual(len(content), 1)
                    context = json.loads(content[0]['text'])
                else:
                    self.assertEqual(generate.call_args.args[1], [])
                    context = json.loads(generate.call_args.args[0].split('分段素材包：\n')[1])
                self.assertTrue(all('sample_timestamps_seconds' not in video for video in context['videos']))
                self.assertTrue(all('未观察视频' in video['observation'] for video in context['videos']))

    def test_each_effective_video_field_invalidates_both_provider_keys(self):
        changes = dict(video_note='@视频1参考横移，@视频2参考动作', video_reference_source='custom',
            video_paths=['two.mp4', 'one.mp4'], video_hashes=['changed', 'video-hash-two'],
            video_refs=['video-two', 'video-one'], video_durations=[3.5, 4.0])
        for provider, profile in [('api', PROFILE), ('codex_cli', CLI_PROFILE)]:
            key = expansion.cache_key(self.packet, 'vision', 'fixture', '规则', 0, profile=profile, provider=provider)
            for field, changed in changes.items():
                with self.subTest(provider=provider, field=field):
                    other = {**self.packet, field: changed}
                    self.assertNotEqual(key, expansion.cache_key(other, 'vision', 'fixture', '规则', 0, profile=profile, provider=provider))

    def test_legacy_no_video_keys_and_cache_are_unchanged_by_empty_video_fields(self):
        packet = {name: value for name, value in self.packet.items() if name not in expansion.VIDEO_FIELDS}
        key = expansion.cache_key(packet, 'text', 'fixture', '规则', 0, profile=PROFILE)
        context = {name: packet[name] for name in ('hashes', 'material_note', 'visual_type', 'mode', 'brief',
            'duration', 'generation_frames', 'generation_seconds', 'audio_role', 'audio_section', 'audio_role_reason')}
        context.update(material_note=expansion.model_material_note(packet['material_note'], 2),
            audio_reference_available=False, audio_reference_slot=None)
        expected = hashlib.sha256(json.dumps([context, 'text', 'fixture', '规则', 0, PROFILE['base_url'], expansion.GUARD], sort_keys=True).encode()).hexdigest()
        self.assertEqual(key, expected)
        empty = dict(video_refs=[], video_paths=[], video_hashes=[], video_durations=[], video_note='', video_reference_source='default')
        self.assertEqual(key, expansion.cache_key({**packet, **empty}, 'text', 'fixture', '规则', 0, profile=PROFILE))
        cache = Path(packet['cache_dir'])
        cache.mkdir()
        (cache/(key+'.json')).write_text(json.dumps({'text': fixtures.TEXT}), encoding='utf-8')
        with patch.object(expansion, 'call', side_effect=AssertionError('Legacy cache not used')):
            self.assertEqual(self.expand(packet, mode='text'), fixtures.TEXT)

    def test_video_invalid_saved_cache_and_new_output_are_all_validated(self):
        packet = self.one_video()
        key = expansion.cache_key(packet, 'text', 'fixture', '规则', 0, profile=PROFILE)
        cache = Path(packet['cache_dir'])
        cache.mkdir()
        (cache/(key+'.json')).write_text(json.dumps({'text': fixtures.TEXT}), encoding='utf-8')
        packet.update(expanded_prompt=fixtures.TEXT, expanded_key=key)
        self.assertEqual(expansion.cache_source(packet, key, 'text'), 'api')
        with patch.object(expansion, 'call', return_value={'choices': [{'message': {'content': video_prompt()}}]}) as provider:
            self.assertEqual(self.expand(packet, mode='text'), video_prompt())
        provider.assert_called_once()
        self.assertEqual(json.loads((cache/(key+'.json')).read_text(encoding='utf-8'))['text'], video_prompt())
        other = {**packet, 'video_hashes': ['changed'], 'expanded_key': ''}
        with patch.object(expansion, 'call', return_value={'choices': [{'message': {'content': fixtures.TEXT}}]}) as provider:
            with self.assertRaises(ValueError):
                self.expand(other, mode='text')
        provider.assert_called_once()
        self.assertEqual(len(list(cache.glob('*.json'))), 1)

    def test_batch_freezes_every_video_field_and_rejects_invalid_metadata(self):
        packet = copy.deepcopy(self.packet)
        frozen = {name: copy.deepcopy(packet[name]) for name in expansion.VIDEO_FIELDS}
        settings = dict(mode='text', model='fixture', rule='规则', revision=0, provider='api')
        job = jobs.start(self.directory/'project', 'video-project', 1,
            [dict(index=0, material=packet, protected=True)], settings, request_profile=PROFILE)
        packet['video_paths'].reverse()
        packet['video_hashes'][0] = 'changed'
        packet['video_note'] = 'changed'
        material = jobs.get(self.directory/'project', job['id'])['items'][0]['material']
        self.assertEqual({name: material[name] for name in expansion.VIDEO_FIELDS}, frozen)
        bad = self.one_video()
        bad.pop('video_durations')
        with patch.object(expansion, 'call') as provider:
            job = jobs.start(self.directory/'invalid', 'invalid-video-project', 1,
                [dict(index=0, material=bad)], settings, request_profile=PROFILE)
        self.assertEqual(job['counts'], {'failed': 1})
        provider.assert_not_called()

    def test_manual_direct_output_remains_unvalidated(self):
        packet = {**self.packet, 'brief': '手写原文直接输出。', 'video_durations': [], 'generation_seconds': 99}
        with patch.object(expansion, 'call', side_effect=AssertionError('Manual provider call')):
            self.assertEqual(self.expand(packet, mode='manual'), packet['brief'])


if __name__ == '__main__':
    unittest.main()
