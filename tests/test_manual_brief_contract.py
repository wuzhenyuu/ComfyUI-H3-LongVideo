"""Human director edits survive expansion beside unified requests and music analysis."""
import copy
import importlib
import json
import unittest
from unittest.mock import patch

import test_library_batch_routes as route_fixtures
import test_materials as fixtures
from test_expansion_fixtures import chinese_prompt
from test_music_emotion_expansion import ANALYSIS, PROFILE

core, materials, expansion = fixtures.core, fixtures.materials, fixtures.expansion
cli = importlib.import_module('h3lv_test.codex_cli')
MANUAL = ('  人工导演：人物先停顿，再抬眼；保留蓝色背景。\n'
          '音乐表达（人工修订）：此处应当克制而期待，不能按模型的平静估计处理。\n')


class ManualBriefContractTests(unittest.TestCase):
    setUp = fixtures.MaterialTests.setUp

    def packet(self, source='user'):
        row = self.plan['segments'][0]
        row.update(shot_request='@图1 是人物，@图2 是场景。人物向左走，镜头跟随。',
                   prompt=MANUAL, brief_source=source,
                   music_emotion={'status': 'ready', 'analysis': copy.deepcopy(ANALYSIS)})
        return materials.packet(self.plan, row, self.directory)

    def test_packet_preserves_entire_human_brief_without_changing_automatic_fallback(self):
        packet = self.packet()
        self.assertEqual(packet['user_director_brief'], MANUAL)
        self.assertEqual(packet['brief'], '')
        self.assertEqual(packet['brief_source'], 'none')
        self.assertEqual(packet['music_emotion'], ANALYSIS)
        row = self.plan['segments'][0]
        row['shot_request'] = ''
        packet = materials.packet(self.plan, row, self.directory)
        self.assertEqual(packet['user_director_brief'], MANUAL)
        self.assertEqual(packet['brief_source'], 'automatic')
        self.assertTrue(packet['brief'])

    def test_automatic_legacy_and_empty_briefs_do_not_claim_human_provenance(self):
        for source in ('automatic', 'legacy', None):
            with self.subTest(source=source):
                packet = self.packet(source)
                self.assertNotIn('user_director_brief', packet)
                self.assertNotIn('user_director_brief', expansion.intent_context(packet))
                self.assertNotIn(expansion.USER_DIRECTOR_GUARD, expansion.generation_guard(packet))
        row = self.plan['segments'][0]
        row.update(brief_source='user', prompt=' \n\t')
        self.assertNotIn('user_director_brief', materials.packet(self.plan, row, self.directory))

    def test_edit_roundtrip_marks_only_edited_segment_as_user_and_preserves_final_text(self):
        originals = copy.deepcopy(self.plan['segments'])
        updates = copy.deepcopy(originals)
        updates[0].update(prompt=MANUAL, brief_source='user')
        self.plan['segments'][0]['final_prompt'] = 'FINAL EXACT\n手工最终正文。'
        core.edit_plan(self.plan, updates, self.directory)
        self.assertEqual(self.plan['changed_segments'], [0])
        self.assertEqual(self.plan['segments'][0]['prompt'], MANUAL)
        self.assertEqual(self.plan['segments'][0]['brief_source'], 'user')
        self.assertEqual(self.plan['segments'][0]['final_prompt'], 'FINAL EXACT\n手工最终正文。')
        for current, previous in zip(self.plan['segments'][1:], originals[1:]):
            self.assertEqual(current['prompt'], previous['prompt'])
            self.assertEqual(current['brief_source'], previous['brief_source'])
        root = self.directory/'projects'
        core.write_plan(root, self.plan)
        saved = core.read_plan(root, self.plan['id'])
        self.assertEqual(saved['segments'][0]['brief_source'], 'user')
        self.assertEqual(materials.packet(saved, saved['segments'][0], self.directory)['user_director_brief'], MANUAL)

    def test_automatic_music_annotation_does_not_become_a_human_edit(self):
        from h3lv_test.music_emotion import annotate_brief
        row = self.plan['segments'][0]
        row['music_emotion'] = {'status': 'ready', 'analysis': copy.deepcopy(ANALYSIS)}
        updates = copy.deepcopy(self.plan['segments'])
        updates[0].update(prompt=annotate_brief(row['prompt'], ANALYSIS), brief_source='automatic')
        core.edit_plan(self.plan, updates, self.directory)
        self.assertEqual(row['brief_source'], 'automatic')
        self.assertNotIn('user_director_brief', materials.packet(self.plan, row, self.directory))

    def test_source_only_human_apply_invalidates_only_that_take_and_preserves_originals(self):
        baseline = copy.deepcopy(self.plan)
        for segment in baseline['segments']:
            segment.setdefault('reference_source', 'default')
        for source in ('automatic', 'legacy', None):
            with self.subTest(source=source):
                plan = copy.deepcopy(baseline)
                row = plan['segments'][0]
                row.update(prompt=MANUAL, shot_request='人物向左走，镜头跟随。',
                           final_prompt='FINAL EXACT\n保持最终手写原文。',
                           job={'status': 'completed', 'video': 'accepted-old.mp4', 'prompt_id': 'old'})
                if source is None:
                    row.pop('brief_source', None)
                else:
                    row['brief_source'] = source
                previous_take = copy.deepcopy(row['job'])
                other_segments = copy.deepcopy(plan['segments'][1:])
                before = expansion.cache_key(materials.packet(plan, row, self.directory),
                    'text', 'fixture', '', 0, profile=PROFILE)
                updates = copy.deepcopy(plan['segments'])
                updates[0]['brief_source'] = 'user'
                core.edit_plan(plan, updates, self.directory)
                self.assertEqual(plan['changed_segments'], [0])
                self.assertEqual(row['brief_source'], 'user')
                self.assertEqual(row['prompt'], MANUAL)
                self.assertEqual(row['final_prompt'], 'FINAL EXACT\n保持最终手写原文。')
                self.assertEqual(row['job'], previous_take)
                self.assertTrue(row['needs_regeneration'])
                self.assertEqual(plan['segments'][1:], other_segments)
                packet = materials.packet(plan, row, self.directory)
                self.assertEqual(packet['user_director_brief'], MANUAL)
                self.assertNotEqual(before, expansion.cache_key(packet, 'text', 'fixture', '', 0, profile=PROFILE))
                root = self.directory/'projects'
                core.write_plan(root, plan)
                self.assertEqual(core.read_plan(root, plan['id'])['segments'][0]['brief_source'], 'user')
                nonce = row['regeneration_nonce']
                core.edit_plan(plan, copy.deepcopy(plan['segments']), self.directory)
                self.assertEqual(plan['changed_segments'], [])
                self.assertEqual(row['regeneration_nonce'], nonce)

    def test_api_and_cli_receive_original_human_brief_music_and_unified_request(self):
        for provider in ('api', 'codex_cli'):
            for mode in ('text', 'vision'):
                with self.subTest(provider=provider, mode=mode):
                    packet = self.packet()
                    profile = PROFILE if provider == 'api' else {'cli_identity': {'backend': 'fixture'}}
                    with patch.object(expansion, 'call', return_value={'choices': [{'message': {'content': chinese_prompt()}}]}) as api, \
                         patch.object(cli, 'generate', return_value=chinese_prompt()) as codex:
                        expansion.expand(packet, mode, 'fixture', 'style', provider=provider,
                            request_profile=profile, cancelled=lambda: False)
                    if provider == 'api':
                        payload = api.call_args.args[1]
                        context = json.loads(payload['messages'][1]['content'][0]['text'])
                        guard = payload['messages'][0]['content']
                    else:
                        guard = codex.call_args.args[0]
                        context = json.loads(guard.split('分段素材包：\n')[1])
                    self.assertEqual(context['user_director_brief'], MANUAL)
                    self.assertEqual(context['music_emotion'], ANALYSIS)
                    self.assertIn('人物向左走，镜头跟随', context['material_note'])
                    self.assertEqual(context['brief'], '')
                    self.assertIs(context['audio_reference_available'], False)
                    self.assertIn('not an automatic fallback', guard)
                    self.assertIn('take precedence for the same field', guard)
                    self.assertIn('human correction to music analysis takes precedence', guard)

    def test_human_edit_invalidates_api_and_cli_keys_without_reusing_saved_output(self):
        packet = self.packet()
        for provider in ('api', 'codex_cli'):
            profile = PROFILE if provider == 'api' else {'cli_identity': {'backend': 'fixture'}}
            selected = dict(mode='text', model='fixture', rule='style', revision=0, provider=provider)
            original = expansion.cache_key(packet, **selected, profile=profile)
            changed = copy.deepcopy(packet)
            changed.update(user_director_brief=MANUAL+'新的人工情绪修订。',
                           expanded_key=original, expanded_prompt=chinese_prompt())
            with self.subTest(provider=provider):
                self.assertNotEqual(original, expansion.cache_key(changed, **selected, profile=profile))
                with patch.object(expansion, 'call', return_value={'choices': [{'message': {'content': chinese_prompt()}}]}) as api, \
                     patch.object(cli, 'generate', return_value=chinese_prompt()) as codex:
                    expansion.expand(changed, **selected, request_profile=profile, cancelled=lambda: False)
                (api if provider == 'api' else codex).assert_called_once()

    def test_manual_mode_combines_human_condition_and_keeps_missing_field_behavior(self):
        packet = self.packet()
        with patch.object(expansion, 'call') as provider:
            self.assertEqual(expansion.expand(packet, 'manual', 'fixture', '', request_profile=PROFILE),
                             packet['material_note']+'\n\n'+MANUAL)
            packet.pop('user_director_brief')
            self.assertEqual(expansion.expand(packet, 'manual', 'fixture', '', request_profile=PROFILE),
                             packet['material_note'])
            packet.update(contract_version=1, brief='legacy fallback')
            self.assertEqual(expansion.expand(packet, 'manual', 'fixture', '', request_profile=PROFILE),
                             'legacy fallback')
        provider.assert_not_called()


class ManualBriefRouteTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = route_fixtures.LibraryBatchRouteTests.asyncSetUp
    asyncTearDown = route_fixtures.LibraryBatchRouteTests.asyncTearDown
    wait_job = route_fixtures.LibraryBatchRouteTests.wait_job
    start = route_fixtures.LibraryBatchRouteTests.start

    def prepare_human_briefs(self):
        for row in self.plan['segments'][:2]:
            row.update(prompt=MANUAL, brief_source='user',
                       music_emotion={'status': 'ready', 'analysis': copy.deepcopy(ANALYSIS)})
        core.write_plan(self.root, self.plan)

    async def test_batch_freeze_provider_and_apply_keep_human_condition_and_key(self):
        self.prepare_human_briefs()
        with patch.object(expansion, 'call', return_value={'choices': [{'message': {'content': fixtures.TEXT}}]}) as provider:
            job = await self.start({**self.payload, 'indices': [0]})
            result = await self.wait_job(job['id'])
        self.assertEqual(result['counts'], {'succeeded': 1})
        item = result['items'][0]
        self.assertEqual(item['material']['user_director_brief'], MANUAL)
        context = json.loads(provider.call_args.args[1]['messages'][1]['content'][0]['text'])
        self.assertEqual(context['user_director_brief'], MANUAL)
        self.assertEqual(context['music_emotion'], ANALYSIS)
        current = route_fixtures.routes.batch_items(core.read_plan(self.root, self.plan['id']), self.project, [0])[0]['material']
        profile = route_fixtures.routes.expansion_profile(result['settings'])
        self.assertEqual(item['key'], expansion.cache_key(current, **result['settings'], profile=profile))
        response = await self.client.post(self.base+'/expansion-batches/'+job['id']+'/apply',
            json={'revision': 1, 'indices': [0]})
        self.assertEqual(response.status, 200, await response.text())
        saved = core.read_plan(self.root, self.plan['id'])['segments'][0]
        self.assertEqual(saved['expanded_key'], item['key'])
        self.assertEqual(saved['prompt'], MANUAL)
        self.assertEqual(saved['brief_source'], 'user')

    async def test_batch_apply_rejects_a_human_condition_changed_since_generation(self):
        self.prepare_human_briefs()
        with patch.object(expansion, 'call', return_value={'choices': [{'message': {'content': fixtures.TEXT}}]}):
            job = await self.start({**self.payload, 'indices': [0]})
            await self.wait_job(job['id'])
        changed = core.read_plan(self.root, self.plan['id'])
        changed['segments'][0]['prompt'] += '新的人工导演要求。'
        core.write_plan(self.root, changed)
        response = await self.client.post(self.base+'/expansion-batches/'+job['id']+'/apply',
            json={'revision': 1, 'indices': [0]})
        self.assertEqual(response.status, 400)
        self.assertIn('简报或扩写设置已变化', (await response.json())['error'])
        saved = core.read_plan(self.root, self.plan['id'])['segments'][0]
        self.assertEqual(saved['prompt'], changed['segments'][0]['prompt'])
        self.assertNotIn('expanded_prompt', saved)

    async def test_human_expanded_and_final_prompts_remain_protected(self):
        self.prepare_human_briefs()
        self.plan['segments'][0]['expanded_prompt'] = '人工扩写原文\n精确保留。'
        self.plan['segments'][1]['final_prompt'] = 'FINAL EXACT\n最终手写。'
        core.write_plan(self.root, self.plan)
        with patch.object(expansion, 'call') as provider:
            job = await self.start({**self.payload, 'indices': [0, 1]})
            result = await self.wait_job(job['id'])
        provider.assert_not_called()
        self.assertEqual(result['counts'], {'protected': 2})
        saved = core.read_plan(self.root, self.plan['id'])
        self.assertEqual(saved['segments'][0]['expanded_prompt'], '人工扩写原文\n精确保留。')
        self.assertEqual(saved['segments'][1]['final_prompt'], 'FINAL EXACT\n最终手写。')


if __name__ == '__main__':
    unittest.main()
