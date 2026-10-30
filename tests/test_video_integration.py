import copy
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import test_materials as fixtures
from test_video_materials import make_video

core, nodes, controller, materials = fixtures.core, fixtures.nodes, fixtures.controller, fixtures.materials
videos = importlib.import_module('h3lv_test.video_materials')


class VideoIntegrationTests(unittest.TestCase):
    setUp = fixtures.MaterialTests.setUp
    updates = fixtures.MaterialTests.updates

    @classmethod
    def setUpClass(cls):
        cls.media_tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.media_tmp.cleanup)
        folder = Path(cls.media_tmp.name)
        cls.first = make_video(folder/'first.mp4', 2).read_bytes()
        cls.second = make_video(folder/'second.mp4', 3, color='blue').read_bytes()
        cls.eight = make_video(folder/'eight.mp4', 8).read_bytes()

    def store(self, content, name='source.mp4'):
        return videos.store_reference(self.directory, name, content)['name']

    def test_empty_old_project_keeps_fingerprints_and_external_video_connections(self):
        before = core.fingerprint(self.plan)
        self.assertFalse(materials.videos_enabled(self.plan))
        prompt = self.prompt()
        original = prompt['2']['inputs']['ref_videos.ref_video_0']
        controller.apply_segment_references(prompt, self.plan, self.plan['segments'][0], self.directory)
        self.assertEqual(prompt['2']['inputs']['ref_videos.ref_video_0'], original)
        core.edit_plan(self.plan, self.updates(), self.directory)
        self.assertEqual(core.fingerprint(self.plan), before)
        self.assertNotIn('video_materials_version', self.plan)
        self.assertNotIn('video_paths', materials.packet(self.plan, self.plan['segments'][0], self.directory))

    def test_default_inheritance_custom_override_and_only_affected_takes_stale(self):
        first, second = self.store(self.first), self.store(self.second)
        changed = self.updates()
        changed[1].update(video_reference_source='custom', video_refs=[second], video_note='@视频1 仅参考动作')
        core.edit_plan(self.plan, changed, self.directory,
            materials={'refs': self.plan['default_refs'], 'note': self.plan['default_material_note'],
                       'video_refs': [first], 'video_note': '@视频1 仅参考运镜'})
        for index, row in enumerate(self.plan['segments']):
            row['job'] = {'status': 'completed', 'video': f'old-{index}.mp4'}
        self.assertEqual(videos.effective(self.plan, self.plan['segments'][0])['refs'], [first])
        self.assertEqual(videos.effective(self.plan, self.plan['segments'][1])['refs'], [second])
        core.edit_plan(self.plan, self.updates(), self.directory,
            materials={'refs': self.plan['default_refs'], 'note': self.plan['default_material_note'],
                       'video_refs': [second], 'video_note': '@视频1 新运镜'})
        self.assertEqual(self.plan['changed_segments'], [0, 2])
        self.assertNotIn('needs_regeneration', self.plan['segments'][1])
        self.assertEqual(self.plan['segments'][0]['job']['video'], 'old-0.mp4')

    def test_one_custom_video_does_not_invalidate_other_empty_segments(self):
        changed = self.updates()
        changed[1].update(video_reference_source='custom', video_refs=[self.store(self.first)], video_note='动作')
        core.edit_plan(self.plan, changed, self.directory)
        self.assertEqual(self.plan['changed_segments'], [1])

    def test_legacy_canvas_materials_require_explicit_opt_in(self):
        self.plan.pop('materials_version')
        changed = self.updates()
        changed[0].update(video_reference_source='custom', video_refs=[self.store(self.first)])
        before = copy.deepcopy(self.plan)
        with self.assertRaisesRegex(ValueError, '启用内置素材管理'):
            core.edit_plan(self.plan, changed, self.directory)
        self.assertEqual(self.plan, before)

    def test_invalid_group_or_reference_is_rejected_before_plan_mutation(self):
        a, b = self.store(self.eight), self.store(self.eight)
        for note, names in [('', [a, b]), ('@视频2 运镜', [a])]:
            before = copy.deepcopy(self.plan)
            with self.assertRaises(ValueError):
                core.edit_plan(self.plan, self.updates(), self.directory,
                    materials={'refs': self.plan['default_refs'], 'note': '', 'video_refs': names, 'video_note': note})
            self.assertEqual(self.plan, before)

    def test_generation_preflight_checks_only_target_video_materials(self):
        name = self.store(self.first)
        self.plan.update(video_materials_version=1, default_video_refs=[name], default_video_note='')
        self.plan['segments'][1].update(video_reference_source='custom', video_refs=[], video_note='')
        videos.path(self.directory, name).unlink()
        with self.assertRaisesRegex(ValueError, '第 1 段'):
            controller.validate_generation_materials(self.plan, self.directory, [0])
        controller.validate_generation_materials(self.plan, self.directory, [1])

    @staticmethod
    def prompt():
        return {'1': {'class_type': 'H3LVUnified', 'inputs': {}},
                '2': {'class_type': 'MiniMaxH3ReferenceToVideo', 'inputs': {
                    'prompt': 'keep', 'ref_videos.ref_video_0': ['external', 0],
                    'ref_videos.ref_video_2': ['external', 2],
                    'ref_video_audios.ref_video_audio_0': ['external', 1],
                    'ref_audios.ref_audio_0': ['1', 1]}}}

    def test_controller_binds_ordered_new_slots_and_only_removes_video_soundtracks(self):
        names = [self.store(self.first), self.store(self.second)]
        self.plan.update(video_materials_version=1, default_video_refs=names, default_video_note='')
        prompt = self.prompt()
        controller.apply_segment_references(prompt, self.plan, self.plan['segments'][0], self.directory)
        inputs = prompt['2']['inputs']
        self.assertEqual(inputs['ref_videos.ref_video_0'], ['1', 13])
        self.assertEqual(inputs['ref_videos.ref_video_1'], ['1', 14])
        self.assertNotIn('ref_videos.ref_video_2', inputs)
        self.assertNotIn('ref_video_audios.ref_video_audio_0', inputs)
        self.assertEqual(inputs['ref_audios.ref_audio_0'], ['1', 1])
        self.plan['segments'][0].update(video_reference_source='custom', video_refs=[], video_note='')
        controller.apply_segment_references(prompt, self.plan, self.plan['segments'][0], self.directory)
        self.assertFalse(any(name.startswith('ref_videos.') for name in inputs))

    def test_node_appends_real_image_frames_without_changing_previous_outputs(self):
        self.assertEqual(nodes.LoadSegment.RETURN_NAMES[:13], (
            'original_audio_padded', 'vocals_padded', 'generation_frames', 'filename_prefix', 'segment_material',
            'image_1', 'image_2', 'image_3', 'image_4', 'image_5', 'image_6', 'segment_prompt', 'fps'))
        self.assertEqual(nodes.Unified.RETURN_NAMES[13:], ('video_1', 'video_2', 'video_3'))
        self.assertEqual(nodes.Unified.RETURN_TYPES[13:], ('IMAGE',)*3)
        self.plan.update(video_materials_version=1, default_video_refs=[self.store(self.first)], default_video_note='运镜')
        self.plan['approved'] = True
        self.plan['approved_fingerprint'] = core.fingerprint(self.plan)
        audio = np.zeros((10000, 1), dtype=np.float32)
        for name in ('source.wav', 'vocals.wav'):
            target = core.audio_file(self.directory, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, audio, 100)
        with patch.object(nodes, 'data_root', return_value=self.directory), \
                patch.object(nodes, 'read_plan', return_value=self.plan), \
                patch.object(nodes, 'project_path', return_value=self.directory):
            result = nodes.LoadSegment().load(self.plan['id'], 0)
        self.assertEqual(len(result), 16)
        self.assertEqual(result[12], 24.0)
        self.assertEqual(tuple(result[13].shape), (48, 48, 64, 3))
        self.assertIsNone(result[14])
        self.assertIsNone(result[15])


if __name__ == '__main__':
    unittest.main()
