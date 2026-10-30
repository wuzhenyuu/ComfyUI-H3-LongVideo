"""Offline RH compatibility for local expansion providers and video references."""
import copy
import importlib
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import torch
import test_materials as fixtures
import test_mv_rh as cloud
from test_expansion_fixtures import chinese_prompt
from test_manual_brief_contract import MANUAL
from test_music_emotion_expansion import ANALYSIS

remote, expansion = cloud.remote, fixtures.expansion
core, controller = cloud.core, cloud.controller
cli = importlib.import_module('h3lv_test.codex_cli')


def payload(**settings):
    value = cloud.payload()
    value['prompt']['2']['inputs']['prompt'] = ['4', 0]
    value['prompt']['4'] = {'class_type': 'H3LVPromptExpand', 'inputs': {
        'material': ['1', 4], 'mode': 'text', 'model': 'fixture-api-model',
        'rule': 'fixture rule', 'revision': 2, **settings}}
    return value


class RHExpansionCompatibilityTests(unittest.TestCase):
    def setUp(self):
        fixtures.MaterialTests.setUp(self)
        self.plan['segments'][0].update(visual_type='atmosphere',
            shot_request='@图1 是人物，@图2 是环境。人物向左走，镜头跟随。',
            prompt=MANUAL, brief_source='user',
            music_emotion={'status': 'ready', 'analysis': copy.deepcopy(ANALYSIS)})
        self.material = fixtures.materials.packet(self.plan, self.plan['segments'][0], self.directory)
        self.identity = patch.object(cli, 'identity', side_effect=lambda model='', reasoning='default': {
            'backend': 'codex_cli_v1', 'config_hash': 'fixture-config-hash',
            'model': model, 'reasoning': reasoning})
        self.identity.start()
        self.addCleanup(self.identity.stop)
        interrupted = patch.object(expansion, '_codex_cancelled', return_value=None)
        interrupted.start()
        self.addCleanup(interrupted.stop)

    def snapshot(self, **settings):
        value = remote.capture(payload(**settings))
        value.update(remote_graph=cloud.graph(), binding=remote.binding(cloud.graph()))
        return value

    def build(self, snapshot, material=None):
        material = self.material if material is None else material
        wave = {'waveform': torch.zeros(1, 1, 1600), 'sample_rate': 16000}
        loaded = (wave, wave, 242, 'prefix', material, *([None] * 6), 'manual', 24, None, None, None)
        folder = self.directory/'take'
        folder.mkdir(exist_ok=True)
        client = MagicMock()
        client.upload.side_effect = lambda path: Path(path).name
        with patch.object(cloud.nodes.LoadSegment, 'load', return_value=loaded), \
                patch('h3lv_test.progress.report'):
            result = remote.build_segment(self.directory, self.plan['id'], 0, snapshot, client, folder)
        return result, client

    def test_capture_freezes_complete_options_and_only_public_cli_identity(self):
        snapshot = self.snapshot(provider='codex_cli', codex_model='fixture-cli-model',
                                 codex_reasoning='xhigh', codex_timeout=600)
        source = snapshot['prompt_source']
        self.assertEqual(source['settings'], dict(mode='text', model='fixture-api-model',
            rule='fixture rule', revision=2, provider='codex_cli', codex_model='fixture-cli-model',
            codex_reasoning='xhigh', codex_timeout=600))
        self.assertEqual(set(source['profile']), {'cli_identity'})
        self.assertEqual(set(source['profile']['cli_identity']), {'backend', 'config_hash', 'model', 'reasoning'})
        self.assertNotIn('api_key', json.dumps(snapshot))
        self.assertNotIn('config.toml', json.dumps(snapshot))
        api = self.snapshot()['prompt_source']
        self.assertEqual({key: api['settings'][key] for key in expansion.generation_options()},
                         expansion.generation_options())
        self.assertNotIn('profile', api)

    def test_cli_build_calls_cli_only_with_frozen_options_and_shared_material_context(self):
        snapshot = self.snapshot(provider='codex_cli', codex_model='fixture-cli-model',
                                 codex_reasoning='high', codex_timeout=601)
        with patch.object(cli, 'identity', side_effect=AssertionError('must use frozen identity')), \
                patch.object(cli, 'generate', return_value=chinese_prompt()) as generate, \
                patch.object(expansion, 'public_settings', side_effect=AssertionError('API profile read')), \
                patch.object(expansion, 'call', side_effect=AssertionError('API called')) as api:
            (graph, text), client = self.build(snapshot)
            self.build(snapshot)
        self.assertEqual(text, chinese_prompt())
        self.assertEqual(graph['136']['inputs']['prompt'], text)
        self.assertNotIn('ref_audios.ref_audio_0', graph['136']['inputs'])
        api.assert_not_called()
        generate.assert_called_once()
        self.assertEqual(generate.call_args.kwargs['model'], 'fixture-cli-model')
        self.assertEqual(generate.call_args.kwargs['reasoning'], 'high')
        self.assertEqual(generate.call_args.kwargs['timeout'], 601)
        self.assertEqual(generate.call_args.kwargs['expected_identity'], snapshot['prompt_source']['profile']['cli_identity'])
        context = json.loads(generate.call_args.args[0].split('分段素材包：\n')[1])
        self.assertEqual(context['user_director_brief'], MANUAL)
        self.assertEqual(context['music_emotion'], ANALYSIS)
        self.assertEqual(context['brief'], '')
        self.assertIn('人物向左走', context['material_note'])
        self.assertEqual(context['audio_reference_available'], False)
        self.assertEqual(generate.call_args.args[1], [])
        self.assertEqual(len(client.upload.call_args_list), 4)
        self.assertNotIn('audio_reference_available', self.material)
        cached = list(Path(self.material['cache_dir']).glob('*.json'))
        self.assertEqual(len(cached), 1)
        self.assertEqual(json.loads(cached[0].read_text(encoding='utf-8'))['provider'], 'codex_cli')

    def test_old_and_explicit_api_snapshots_keep_the_legacy_call_contract(self):
        snapshots = [self.snapshot(), self.snapshot(provider='api', codex_model='unused', codex_timeout=999)]
        old = copy.deepcopy(snapshots[0])
        for key in expansion.generation_options():
            old['prompt_source']['settings'].pop(key)
        snapshots.append(old)
        for snapshot in snapshots:
            with self.subTest(source=snapshot['prompt_source']), \
                    patch.object(expansion.PromptExpand, 'run', return_value={'result': (chinese_prompt(),)}) as run, \
                    patch.object(cli, 'generate') as generate:
                self.build(snapshot)
            self.assertEqual(run.call_args.kwargs, dict(mode='text', model='fixture-api-model',
                                                       rule='fixture rule', revision=2))
            generate.assert_not_called()
            self.assertEqual(remote.graph_fingerprint(snapshot), remote.graph_fingerprint(old))

    def test_cli_vision_receives_ordered_images_without_accessing_api(self):
        snapshot = self.snapshot(mode='vision', provider='codex_cli')
        with patch.object(cli, 'generate', return_value=chinese_prompt()) as generate, \
                patch.object(expansion, 'public_settings', side_effect=AssertionError('API profile read')), \
                patch.object(expansion, 'call', side_effect=AssertionError('API called')) as api:
            self.build(snapshot)
        self.assertEqual(generate.call_args.args[1], self.material['paths'])
        self.assertEqual(generate.call_args.kwargs['expected_identity'], snapshot['prompt_source']['profile']['cli_identity'])
        api.assert_not_called()

    def test_provider_and_cli_identity_change_keys_without_changing_timeout_semantics(self):
        snapshot = self.snapshot(provider='codex_cli', codex_model='fixture-cli-model', codex_reasoning='high')
        source = snapshot['prompt_source']
        material = {**self.material, 'audio_reference_available': False, 'audio_reference_slot': None}
        def key(value):
            return expansion.cache_key(material, **value['settings'], profile=value.get('profile'))
        original = key(source)
        changed = copy.deepcopy(source)
        changed['settings']['codex_timeout'] = 600
        self.assertEqual(original, key(changed))
        for field, replacement in [('codex_model', 'another-model'), ('codex_reasoning', 'xhigh')]:
            changed_snapshot = self.snapshot(provider='codex_cli', **{field: replacement})
            self.assertNotEqual(original, key(changed_snapshot['prompt_source']))
            self.assertNotEqual(remote.graph_fingerprint(snapshot), remote.graph_fingerprint(changed_snapshot))
        changed = copy.deepcopy(source)
        changed['profile']['cli_identity']['config_hash'] = 'changed-config'
        self.assertNotEqual(original, key(changed))
        api = self.snapshot()['prompt_source']
        with patch.object(expansion, 'public_settings', return_value={'base_url': 'https://fixture.invalid/v1'}):
            self.assertNotEqual(original, key(api))

    def test_cli_failure_never_falls_back_or_uploads_assets(self):
        snapshot = self.snapshot(provider='codex_cli')
        wave = {'waveform': torch.zeros(1, 1, 1600), 'sample_rate': 16000}
        loaded = (wave, wave, 242, 'prefix', self.material)
        client = MagicMock()
        with patch.object(cloud.nodes.LoadSegment, 'load', return_value=loaded), \
                patch('h3lv_test.progress.report'), \
                patch.object(cli, 'generate', side_effect=ValueError('CLI fixture failed')) as generate, \
                patch.object(expansion, 'call') as api:
            with self.assertRaisesRegex(ValueError, 'CLI fixture failed'):
                remote.build_segment(self.directory, self.plan['id'], 0, snapshot, client, self.directory)
        generate.assert_called_once()
        api.assert_not_called()
        client.upload.assert_not_called()
        client.create.assert_not_called()

    def test_manual_mode_ignores_invalid_unused_cli_options(self):
        with patch.object(cli, 'identity', side_effect=AssertionError('CLI identity read')):
            snapshot = self.snapshot(mode='manual', provider='codex_cli', codex_model='invalid name')
        with patch.object(cli, 'generate') as generate, patch.object(expansion, 'call') as api, \
                patch.object(expansion, 'public_settings', return_value={'base_url': 'https://fixture.invalid/v1'}):
            (_, text), _ = self.build(snapshot)
        self.assertIn(MANUAL, text)
        api.assert_not_called()
        generate.assert_not_called()

    def test_linked_or_invalid_provider_options_fail_before_runtime(self):
        for field, value in [('provider', ['9', 0]), ('provider', 'unknown'),
                             ('codex_timeout', ['9', 0]), ('codex_reasoning', 'unsupported')]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                remote.capture(payload(provider='codex_cli', **{field: value}) if field != 'provider'
                               else payload(**{field: value}))

    def test_video_material_is_rejected_before_expansion_or_upload_for_all_prompt_sources(self):
        for source in ({'kind': 'literal', 'text': 'literal'}, {'kind': 'segment_prompt'},
                       self.snapshot(provider='codex_cli')['prompt_source']):
            snapshot = self.snapshot()
            snapshot['prompt_source'] = source
            material = {**self.material, 'video_paths': ['video.mp4']}
            wave = {'waveform': torch.zeros(1, 1, 1600), 'sample_rate': 16000}
            with self.subTest(kind=source['kind']), \
                    patch.object(cloud.nodes.LoadSegment, 'load', return_value=(wave, wave, 242, '', material)), \
                    patch.object(expansion, 'expand') as expand, patch.object(expansion.PromptExpand, 'run') as run:
                client = MagicMock()
                with self.assertRaisesRegex(ValueError, '参考视频当前仅支持本地生成'):
                    remote.build_segment(self.directory, self.plan['id'], 0, snapshot, client, self.directory)
            expand.assert_not_called()
            run.assert_not_called()
            client.upload.assert_not_called()
            client.create.assert_not_called()
        self.assertEqual(material['video_paths'], ['video.mp4'])

    def test_shipped_cloud_templates_expose_no_connected_reference_video_loader(self):
        for number in ('08', '15'):
            workflow = json.loads(next((Path(__file__).resolve().parents[1]/'examples').glob(number+'*')).read_text(encoding='utf-8'))
            reference = next(node for node in workflow['nodes'] if node['type'] == 'MiniMaxH3ReferenceToVideo')
            slots = [item for item in reference['inputs'] if item['name'].startswith('ref_videos.')]
            with self.subTest(template=number):
                self.assertTrue(slots)
                self.assertTrue(all(item.get('link') is None for item in slots))
                self.assertFalse(any(node['type'] in ('LoadVideo', 'VHS_LoadVideo', 'VHS_LoadVideoPath')
                                     for node in workflow['nodes']))


class RHRecoveryCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    setUp = cloud.RunnerTests.setUp
    tearDown = cloud.RunnerTests.tearDown

    async def test_old_api_pending_snapshot_resumes_with_current_expansion_defaults(self):
        snapshot = remote.capture(payload())
        snapshot.update(remote_graph=cloud.graph(), binding=remote.binding(cloud.graph()))
        for key in expansion.generation_options():
            snapshot['prompt_source']['settings'].pop(key)
        remote.save(self.directory/'state/queue_snapshot.json', snapshot)
        plan = core.read_plan(self.root, self.plan['id'])
        plan['run_only_segment'] = 0
        plan['segments'][0]['job'] = {'backend': 'runninghub', 'status': 'submitted', 'task_id': '555'}
        core.write_plan(self.root, plan)
        with patch.object(controller.asyncio, 'create_task', side_effect=lambda coroutine: (coroutine.close(), MagicMock())[1]) as launch:
            request = payload()
            request['only_segment_index'] = 0
            remote.start(self.root, plan['id'], request, self.server)
        launch.assert_called_once()
        saved = json.loads((self.directory/'state/queue_snapshot.json').read_text(encoding='utf-8'))
        self.assertEqual(saved, snapshot)
        self.assertEqual(core.read_plan(self.root, plan['id'])['segments'][0]['job']['task_id'], '555')

    async def test_video_preflight_preserves_plan_snapshot_and_old_outputs_without_scheduling(self):
        plan = core.read_plan(self.root, self.plan['id'])
        plan.update(materials_version=1, video_materials_version=1, default_video_refs=['motion.mp4'])
        plan['segments'][0].update(needs_regeneration=True,
            job={'status': 'completed', 'video': 'accepted-old.mp4'})
        plan['approved_fingerprint'] = core.fingerprint(plan)
        core.write_plan(self.root, plan)
        before_plan = copy.deepcopy(core.read_plan(self.root, plan['id']))
        before_snapshot = (self.directory/'state/queue_snapshot.json').read_bytes()
        with patch.object(controller.asyncio, 'create_task') as launch, patch.object(remote, 'Client') as client, \
                patch.object(cloud.nodes.LoadSegment, 'load') as load:
            with self.assertRaisesRegex(ValueError, '第 1 段.*本地生成'):
                remote.start(self.root, plan['id'], cloud.payload(), self.server)
        launch.assert_not_called()
        client.assert_not_called()
        load.assert_not_called()
        self.assertEqual(core.read_plan(self.root, plan['id']), before_plan)
        self.assertEqual((self.directory/'state/queue_snapshot.json').read_bytes(), before_snapshot)

    async def test_pending_cli_resume_keeps_frozen_identity_but_rejects_changed_provider_options(self):
        request = payload(provider='codex_cli', codex_model='frozen-model', codex_reasoning='high')
        with patch.object(cli, 'identity', return_value={'config_hash': 'frozen'}):
            snapshot = remote.capture(request)
        snapshot.update(remote_graph=cloud.graph(), binding=remote.binding(cloud.graph()))
        remote.save(self.directory/'state/queue_snapshot.json', snapshot)
        plan = core.read_plan(self.root, self.plan['id'])
        plan['run_only_segment'] = 0
        plan['segments'][0]['job'] = {'backend': 'runninghub', 'status': 'submitted', 'task_id': '555'}
        core.write_plan(self.root, plan)
        with patch.object(cli, 'identity', return_value={'config_hash': 'changed-default-config'}), \
                patch.object(controller.asyncio, 'create_task', side_effect=lambda coroutine: (coroutine.close(), MagicMock())[1]) as launch:
            remote.start(self.root, plan['id'], request, self.server)
        self.assertEqual(json.loads((self.directory/'state/queue_snapshot.json').read_text(encoding='utf-8')), snapshot)
        launch.assert_called_once()
        controller.TASKS.clear()
        changed = copy.deepcopy(request)
        changed['prompt']['4']['inputs']['codex_model'] = 'new-model'
        with patch.object(cli, 'identity', return_value={'config_hash': 'changed-default-config'}), \
                patch.object(controller.asyncio, 'create_task') as launch:
            with self.assertRaisesRegex(ValueError, '未解决'):
                remote.start(self.root, plan['id'], changed, self.server)
        launch.assert_not_called()
        self.assertEqual(core.read_plan(self.root, plan['id'])['segments'][0]['job']['task_id'], '555')

    async def test_video_capability_check_respects_single_segment_and_custom_empty_selection(self):
        plan = core.read_plan(self.root, self.plan['id'])
        plan.update(video_materials_version=1, default_video_refs=['motion.mp4'])
        plan['segments'][1].update(video_reference_source='custom', video_refs=[])
        remote.require_supported_videos(plan, [1])
        with self.assertRaisesRegex(ValueError, '第 1 段'):
            remote.require_supported_videos(plan)
        self.assertEqual(plan['default_video_refs'], ['motion.mp4'])

    async def test_recovered_paid_task_with_video_metadata_keeps_querying_same_task(self):
        plan = core.read_plan(self.root, self.plan['id'])
        folder = self.directory/'paid'
        folder.mkdir()
        remote.save(folder/'submitted_graph.json', cloud.graph())
        plan.update(video_materials_version=1, default_video_refs=['motion.mp4'])
        plan['segments'][0]['job'] = {'backend': 'runninghub', 'status': 'submitted',
            'task_id': '555', 'directory': str(folder), 'final_prompt': 'saved paid prompt'}
        core.write_plan(self.root, plan)
        client = MagicMock()
        client.query.return_value = {'status': 'FAILED'}
        with patch.object(remote, 'build_segment') as build:
            with self.assertRaisesRegex(ValueError, 'RH任务失败'):
                await remote.segment(self.root, plan['id'], 0, self.snapshot, client)
        client.query.assert_called_once_with('555')
        client.create.assert_not_called()
        client.upload.assert_not_called()
        build.assert_not_called()


if __name__ == '__main__':
    unittest.main()
