"""Audio availability comes from verified graph wiring, without GPU or cloud calls."""
import copy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_plugin as support
import test_library_batch_routes as route_support
from test_mv_rh import graph as remote_graph

controller, nodes = support.controller, support.nodes
routes, expansion, jobs = route_support.routes, route_support.expansion, route_support.jobs
remote = importlib.import_module('h3lv_test.mv_rh')
progress = importlib.import_module('h3lv_test.progress')


def local_graph(slot=1):
    return {'1': {'class_type': 'H3LVUnified', 'inputs': {}},
            '2': {'class_type': 'MiniMaxH3ReferenceToVideo',
                  'inputs': {'ref_audios.ref_audio_0': ['1', slot]}}}


class ConfirmedAudioReferenceTests(unittest.TestCase):
    def test_exact_vocal_output_binding_is_required(self):
        self.assertTrue(controller.confirmed_audio_reference(local_graph(), '1'))
        self.assertTrue(controller.confirmed_audio_reference(local_graph(), 1))
        for graph, loader in ((None, '1'), ({}, '1'), (local_graph(0), '1'),
                              (local_graph(True), '1'), (local_graph(1.0), '1'),
                              (local_graph(), 'missing'), ({'1': None}, '1')):
            with self.subTest(graph=graph, loader=loader):
                self.assertFalse(controller.confirmed_audio_reference(graph, loader))
        graph = local_graph()
        graph['1']['class_type'] = 'LoadAudio'
        self.assertFalse(controller.confirmed_audio_reference(graph, '1'))
        graph = local_graph()
        graph['2']['inputs']['ref_audios.ref_audio_0'] = ['other', 1]
        self.assertFalse(controller.confirmed_audio_reference(graph, '1'))

    def test_reference_selection_matches_controller_submission(self):
        graph = local_graph(0)
        graph['3'] = copy.deepcopy(local_graph()['2'])
        self.assertFalse(controller.confirmed_audio_reference(graph, '1'))
        graph['2']['inputs']['ref_audios.ref_audio_0'] = ['1', 1]
        self.assertTrue(controller.confirmed_audio_reference(graph, '1'))


class PreviewAudioReferenceTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = route_support.LibraryBatchRouteTests.asyncSetUp
    asyncTearDown = route_support.LibraryBatchRouteTests.asyncTearDown

    async def test_single_preview_uses_actual_wiring_and_ignores_client_boolean(self):
        cases = [(None, None, False), (local_graph(0), '1', False),
                 (local_graph(), 'other', False), (local_graph(), '1', True)]
        for graph, loader, expected in cases:
            with self.subTest(graph=graph, loader=loader), \
                    patch.object(expansion, 'expand', return_value=route_support.fixtures.TEXT) as expand:
                payload = {**self.payload, 'index': 0, 'audio_reference_available': True,
                           'audio_reference_slot': 1, 'workflow_prompt': graph, 'source_node_id': loader}
                response = await self.client.post(self.base+'/expand', json=payload)
                self.assertEqual(response.status, 200, await response.text())
                material = expand.call_args.args[0]
                self.assertIs(material['audio_reference_available'], expected)
                self.assertEqual(material['audio_reference_slot'], 1 if expected else None)
                self.assertNotIn('workflow_prompt', material)

    async def test_batch_confirmation_is_per_visual_type_and_does_not_persist_graph(self):
        self.plan['segments'][1]['visual_type'] = 'atmosphere'
        support.core.write_plan(self.root, self.plan)
        with patch.object(jobs, 'start', return_value={'id': 'fixture'}) as start:
            response = await self.client.post(self.base+'/expansion-batches', json={
                **self.payload, 'indices': [0, 1], 'workflow_prompt': local_graph(), 'source_node_id': '1'})
        self.assertEqual(response.status, 200, await response.text())
        items = start.call_args.args[3]
        self.assertIs(items[0]['material']['audio_reference_available'], True)
        self.assertEqual(items[0]['material']['audio_reference_slot'], 1)
        self.assertIs(items[1]['material']['audio_reference_available'], False)
        self.assertIsNone(items[1]['material']['audio_reference_slot'])
        for item in items:
            self.assertNotIn('workflow_prompt', item['material'])
            self.assertNotIn('source_node_id', item['material'])


class RunningHubAudioReferenceTests(unittest.TestCase):
    def loaded(self, kind):
        import torch
        audio = {'waveform': torch.ones((1, 1, 100), device='cpu'), 'sample_rate': 100}
        material = {'visual_type': kind, 'paths': [], 'audio_reference_available': True,
                    'audio_reference_slot': 1}
        return (audio, audio, 240, 'prefix', material, *((None,)*6), 'manual', 24., *((None,)*3))

    def test_cloud_binding_is_confirmed_before_expansion_without_mutating_saved_material(self):
        for kind in ('performance', 'atmosphere', 'environment'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                graph = remote_graph()
                snapshot = {'remote_graph': graph, 'binding': remote.binding(graph),
                            'prompt_source': {'kind': 'expansion', 'settings': {}}}
                loaded = self.loaded(kind)
                original = copy.deepcopy(loaded[4])
                client = unittest.mock.MagicMock()
                client.upload.return_value = 'uploaded.wav'
                with patch.object(nodes.LoadSegment, 'load', return_value=loaded), \
                        patch.object(expansion.PromptExpand, 'run', return_value={'result': ('generated prompt',)}) as expand, \
                        patch.object(progress, 'report'), patch('soundfile.write'):
                    submitted, text = remote.build_segment(Path(temporary), 'a'*32, 0, snapshot, client, Path(temporary))
                material = expand.call_args.args[0]
                expected = kind == 'performance'
                self.assertIs(material['audio_reference_available'], expected)
                self.assertEqual(material['audio_reference_slot'], 1 if expected else None)
                self.assertEqual(loaded[4], original)
                self.assertEqual(snapshot['remote_graph'], graph)
                self.assertEqual(text, 'generated prompt')
                remote.binding(submitted, vocal_reference=expected)
                client.create.assert_not_called()

    def test_stale_vocal_mapping_fails_before_expansion_or_upload(self):
        graph = remote_graph()
        mapping = remote.binding(graph)
        mapping['vocals'] = mapping['original']
        snapshot = {'remote_graph': graph, 'binding': mapping,
                    'prompt_source': {'kind': 'expansion', 'settings': {}}}
        client = unittest.mock.MagicMock()
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(nodes.LoadSegment, 'load', return_value=self.loaded('performance')), \
                patch.object(expansion.PromptExpand, 'run') as expand, patch.object(progress, 'report'):
            with self.assertRaisesRegex(ValueError, '人声参考'):
                remote.build_segment(Path(temporary), 'a'*32, 0, snapshot, client, Path(temporary))
        expand.assert_not_called()
        client.upload.assert_not_called()
        client.create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
