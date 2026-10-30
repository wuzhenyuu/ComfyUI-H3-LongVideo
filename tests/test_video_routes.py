import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aiohttp import FormData

import test_library_batch_routes as route_fixtures
from test_video_materials import make_video

routes = importlib.import_module('h3lv_test.routes')
core = importlib.import_module('h3lv_test.core')


class VideoRouteTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = route_fixtures.LibraryBatchRouteTests.asyncSetUp
    asyncTearDown = route_fixtures.LibraryBatchRouteTests.asyncTearDown

    @classmethod
    def setUpClass(cls):
        cls.media_tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.media_tmp.cleanup)
        root = Path(cls.media_tmp.name)
        cls.two = make_video(root/'two.mp4').read_bytes()
        cls.eight = make_video(root/'eight.mp4', 8).read_bytes()

    async def upload_video(self, path=None, content=None, **fields):
        form = FormData()
        form.add_field('video', self.two if content is None else content,
                       filename='reference.mp4', content_type='video/mp4')
        for name, value in fields.items():
            form.add_field(name, str(value))
        return await self.client.post(path or self.base+'/video-refs', data=form)

    async def test_project_upload_metadata_and_range_preview_leave_plan_unchanged(self):
        response = await self.upload_video()
        self.assertEqual(response.status, 200, await response.text())
        uploaded = await response.json()
        self.assertEqual(uploaded['metadata']['duration'], 2)
        name = uploaded['name']
        info = await self.client.get(self.base+'/video-refs/'+name+'/info')
        self.assertEqual(info.status, 200)
        self.assertEqual((await info.json())['duration'], 2)
        preview = await self.client.get(self.base+'/video-refs/'+name, headers={'Range':'bytes=0-63'})
        self.assertEqual(preview.status, 206)
        self.assertEqual(await preview.read(), self.two[:64])
        plan = core.read_plan(self.root, self.plan['id'])
        self.assertEqual(plan['revision'], 1)
        self.assertNotIn('video_materials_version', plan)

    async def test_video_library_import_save_filter_and_soft_delete_preserve_snapshot(self):
        response = await self.upload_video('/h3lv/library/upload', name='运镜', category='video')
        self.assertEqual(response.status, 200, await response.text())
        asset = await response.json()
        video_list = await (await self.client.get('/h3lv/library?media_type=video')).json()
        image_list = await (await self.client.get('/h3lv/library?media_type=image')).json()
        self.assertEqual(len(video_list['assets']), 1)
        self.assertEqual(video_list['assets'][0]['media_type'], 'video')
        self.assertEqual(image_list['assets'], [])
        selected = [{'asset_id':asset['asset']['id'], 'version':1}]
        imported_response = await self.client.post(self.base+'/library/import', json={
            'revision':1, 'selection':selected, 'media_type':'video'})
        self.assertEqual(imported_response.status, 200, await imported_response.text())
        imported = await imported_response.json()
        name = imported['video_refs'][0]
        self.assertEqual(imported['video_metadata'][name]['duration'], 2)
        saved = await self.client.post(self.base+'/edit', json={
            'revision':1, 'segments':self.plan['segments'], 'materials':{
                'refs':self.plan['default_refs'], 'note':self.plan['default_material_note'],
                'video_refs':[name], 'video_note':'@视频1 仅参考运镜'}})
        self.assertEqual(saved.status, 200, await saved.text())
        plan = await saved.json()
        self.assertEqual(plan['revision'], 2)
        self.assertEqual(plan['video_materials_version'], 1)
        deleted = await self.client.post('/h3lv/library/'+asset['asset']['id']+'/delete',
            json={'revision':asset['revision']})
        self.assertEqual(deleted.status, 200)
        preview = await self.client.get(self.base+'/video-refs/'+name)
        self.assertEqual(preview.status, 200)
        self.assertEqual(await preview.read(), self.two)

    async def test_invalid_bytes_and_active_batch_cannot_store_project_video(self):
        invalid = await self.upload_video(content=b'not a video')
        self.assertEqual(invalid.status, 400)
        with patch.object(routes, 'expansion_active', return_value=True):
            busy = await self.upload_video()
        self.assertEqual(busy.status, 400)
        self.assertFalse(list((self.project/'video_refs').glob('*')))

    async def test_combined_duration_rejects_save_atomically(self):
        first = await (await self.upload_video(content=self.eight)).json()
        second = await (await self.upload_video(content=self.eight)).json()
        result = await self.client.post(self.base+'/edit', json={
            'revision':1, 'segments':self.plan['segments'], 'materials':{
                'refs':self.plan['default_refs'], 'note':self.plan['default_material_note'],
                'video_refs':[first['name'], second['name']], 'video_note':''}})
        self.assertEqual(result.status, 400)
        plan = core.read_plan(self.root, self.plan['id'])
        self.assertEqual(plan['revision'], 1)
        self.assertNotIn('default_video_refs', plan)


if __name__ == '__main__':
    unittest.main()
