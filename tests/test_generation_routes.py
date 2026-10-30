from pathlib import Path
import unittest
from unittest.mock import patch

import test_library_batch_routes as fixtures

core, controller, routes = fixtures.core, fixtures.controller, fixtures.routes


class GenerationRouteTests(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = fixtures.LibraryBatchRouteTests.asyncTearDown

    async def asyncSetUp(self):
        await fixtures.LibraryBatchRouteTests.asyncSetUp(self)
        self.plan['approved'] = True
        self.plan['approved_fingerprint'] = core.fingerprint(self.plan)
        core.write_plan(self.root, self.plan)

    async def test_project_capabilities_and_previous_preview_are_read_only(self):
        old = self.project/'old.mp4'
        old.write_bytes(b'old video fixture')
        self.plan['segments'][0]['replacement_previous_job'] = {'status':'completed', 'video':str(old)}
        core.write_plan(self.root, self.plan)
        state = core.state_file(self.project, 'segments.json')
        before = state.read_bytes()
        response = await self.client.get(self.base)
        self.assertEqual(response.status, 200)
        plan = await response.json()
        self.assertEqual(plan['generation_capabilities'], {'resume_unfinished':True, 'single_segment':True})
        self.assertEqual(plan['segments'][0]['video_preview']['filename'], 'old.mp4')
        self.assertNotIn('job', plan['segments'][0])
        self.assertEqual(state.read_bytes(), before)

    async def test_resume_payload_is_forwarded_without_prior_retry_or_plan_write(self):
        payload = {'resume_unfinished':True, 'revision':1, 'prompt':{'fixture':{}}, 'loader_id':'1', 'video_id':'7'}
        state = core.state_file(self.project, 'segments.json')
        before = state.read_bytes()
        with patch.object(controller, 'start') as start:
            response = await self.client.post(self.base+'/run', json=payload)
        self.assertEqual(response.status, 200)
        self.assertEqual(start.call_args.args[:3], (self.root, self.plan['id'], payload))
        self.assertEqual(state.read_bytes(), before)

    async def test_single_route_freezes_exact_index_and_does_not_premark_other_rows(self):
        state = core.state_file(self.project, 'segments.json')
        before = state.read_bytes()
        with patch.object(controller, 'queued_ids', return_value=set()), patch.object(controller, 'start') as start:
            response = await self.client.post(self.base+'/regenerate', json={'index':7, 'revision':1, 'prompt':{}})
        self.assertEqual(response.status, 200, await response.text())
        selected = start.call_args.args[2]
        self.assertEqual(selected['only_segment_index'], 7)
        self.assertTrue(selected['replace_snapshot'])
        self.assertIs(selected['resume_unfinished'], False)
        self.assertEqual(state.read_bytes(), before)

    async def test_invalid_single_requests_do_not_change_plan_or_start(self):
        state = core.state_file(self.project, 'segments.json')
        before = state.read_bytes()
        invalid = [{'index':index, 'revision':1} for index in (-1, 10, True, 1.5, '1', None)]
        invalid += [{'index':0, 'revision':2}, {'index':0, 'revision':True},
                    {'index':0, 'revision':1, 'resume_unfinished':True}]
        with patch.object(controller, 'queued_ids', return_value=set()), patch.object(controller, 'start') as start:
            for payload in invalid:
                response = await self.client.post(self.base+'/regenerate', json=payload)
                self.assertEqual(response.status, 400, await response.text())
            start.assert_not_called()
        self.assertEqual(state.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
