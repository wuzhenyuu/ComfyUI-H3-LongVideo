import asyncio
import copy
import json
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import test_library_batch_routes as fixtures
from test_music_emotion import emotion, Queue, Worker

core, routes, controller = fixtures.core, fixtures.routes, fixtures.controller


class EmotionRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await fixtures.LibraryBatchRouteTests.asyncSetUp(self)
        server = sys.modules['server'].PromptServer.instance
        server.prompt_queue = Queue()
        (self.project/'audio').mkdir(parents=True, exist_ok=True)
        sf.write(self.project/'audio'/'source.wav', np.zeros(self.plan['samples']), self.plan['sample_rate'])
        self.launched, self.release = threading.Event(), threading.Event()
        self.transform, self.returncode, self.requests = None, 0, []
        for context in (patch.object(emotion, 'readiness', return_value=(True, '')),
                        patch.object(emotion, 'model_signature', return_value='fixture-model'),
                        patch.object(emotion, '_prepare_gpu', return_value=True),
                        patch.object(emotion.subprocess, 'Popen', side_effect=lambda *a, **kw: Worker(self, *a, **kw))):
            self.stack.enter_context(context)

    async def asyncTearDown(self):
        self.release.set()
        for _ in range(600):
            if not emotion.active():
                break
            await asyncio.sleep(.01)
        await fixtures.LibraryBatchRouteTests.asyncTearDown(self)

    async def wait_emotion(self):
        for _ in range(600):
            response = await self.client.get(self.base+'/music-emotion')
            value = await response.json()
            if value['job'] and value['job']['status'] != 'running' and not emotion.active():
                return value
            await asyncio.sleep(.01)
        self.fail('Emotion job did not finish')

    async def start_emotion(self):
        response = await self.client.post(self.base+'/music-emotion', json={'revision':1, 'indices':[0, 1]})
        self.assertEqual(response.status, 200, await response.text())
        return await response.json()

    async def test_get_capabilities_and_status_are_read_only(self):
        state = core.state_file(self.project, 'segments.json')
        before = state.read_bytes()
        response = await self.client.get(self.base)
        value = await response.json()
        self.assertEqual(value['music_emotion_capabilities'], {'analyze':True})
        self.assertIs(value['music_emotion_active'], False)
        response = await self.client.get(self.base+'/music-emotion')
        value = await response.json()
        self.assertIs(value['ready'], True)
        self.assertIsNone(value['job'])
        self.assertEqual(value['model_name'], emotion.MODEL_NAME)
        self.assertEqual(state.read_bytes(), before)

    async def test_post_start_background_poll_and_successful_commit(self):
        value = await self.start_emotion()
        self.assertEqual(value['job']['status'], 'running')
        self.assertEqual(value['job']['total'], 2)
        response = await self.client.get(self.base)
        self.assertTrue((await response.json())['music_emotion_active'])
        self.release.set()
        value = await self.wait_emotion()
        self.assertEqual(value['job']['status'], 'completed')
        self.assertEqual(value['job']['done'], 2)
        saved = core.read_plan(self.root, self.plan['id'])
        self.assertEqual(saved['changed_segments'], [0, 1])
        self.assertEqual(saved['revision'], 2)

    async def test_invalid_body_revision_indices_force_do_not_start(self):
        invalid = [{}, {'revision':True, 'indices':[0]}, {'revision':2, 'indices':[0]},
                   {'revision':1, 'indices':[0,0]}, {'revision':1, 'indices':[True]},
                   {'revision':1, 'indices':[0], 'force':'yes'}]
        for value in invalid:
            response = await self.client.post(self.base+'/music-emotion', json=value)
            self.assertEqual(response.status, 400, await response.text())
        self.assertFalse(emotion.active())
        self.assertFalse(self.launched.is_set())

    async def test_active_analysis_blocks_edit_expand_generate_and_delete(self):
        await self.start_emotion()
        with patch.object(controller, 'start') as start:
            for path, value in [('/edit', {'revision':1, 'segments':copy.deepcopy(self.plan['segments'])}),
                                ('/approve', {'revision':1}),
                                ('/expand', {'revision':1, 'index':0}),
                                ('/expansion-batches', self.payload),
                                ('/run', {}), ('/regenerate', {'index':0}), ('/retry', {'index':0}),
                                ('/restore', {'index':0}), ('/assemble', {}), ('/stop', {})]:
                response = await self.client.post(self.base+path, json=value)
                self.assertEqual(response.status, 400, (path, await response.text()))
                self.assertIn('情绪', (await response.json())['error'])
            start.assert_not_called()
        response = await self.client.post('/h3lv/projects/delete', json={'ids':[self.plan['id']]})
        self.assertEqual(response.status, 400)
        self.assertTrue(self.project.is_dir())

    async def test_cancel_returns_state_and_terminates_only_own_worker(self):
        await self.start_emotion()
        for _ in range(100):
            if self.launched.is_set():
                break
            await asyncio.sleep(.01)
        response = await self.client.post(self.base+'/music-emotion/cancel', json={})
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual((await self.wait_emotion())['job']['status'], 'cancelled')
        saved = core.read_plan(self.root, self.plan['id'])
        self.assertEqual(saved['revision'], 1)
        self.assertNotIn('music_emotion', saved['segments'][0])

    async def test_worker_failure_reports_failed_without_fake_results(self):
        self.returncode = 1
        self.transform = lambda value: None
        await self.start_emotion()
        self.release.set()
        result = await self.wait_emotion()
        self.assertEqual(result['job']['status'], 'failed')
        self.assertEqual(core.read_plan(self.root, self.plan['id'])['revision'], 1)

    async def test_missing_model_returns_clear_error_without_job(self):
        with patch.object(emotion, 'readiness', return_value=(False, '模型尚未完整下载')):
            response = await self.client.get(self.base+'/music-emotion')
            self.assertFalse((await response.json())['ready'])
            response = await self.client.post(self.base+'/music-emotion', json={'revision':1, 'indices':[0]})
            self.assertEqual(response.status, 400)
            self.assertIn('模型尚未完整下载', (await response.json())['error'])
        self.assertFalse(emotion.active())

    async def test_lyric_reassociation_removes_stale_emotion_from_saved_brief(self):
        await self.start_emotion()
        self.release.set()
        await self.wait_emotion()
        saved = core.read_plan(self.root, self.plan['id'])
        core.state_file(self.project, 'transcript.json').write_text(json.dumps({'segments':[
            {'start':0, 'end':10, 'text':'新的歌词'}], 'words':[]}), encoding='utf-8')
        response = await self.client.post(self.base+'/edit', json={'revision':saved['revision'], 'segments':saved['segments']})
        self.assertEqual(response.status, 200, await response.text())
        value = await response.json()
        self.assertEqual(value['segments'][0]['music_emotion']['status'], 'stale')
        self.assertNotIn('【音乐情绪分析】', value['segments'][0]['prompt'])


if __name__ == '__main__':
    unittest.main()
