import copy
import importlib
import json
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from test_plugin import core, sample_plan

emotion = importlib.import_module('h3lv_test.music_emotion')
prepare_gpu = emotion._prepare_gpu


def analysis(**changes):
    return dict(labels=['温柔', '期待'], intensity=2.5, trend='由平静逐步增强', summary='轻柔旋律带有期待感',
                melody='旋律舒展', rhythm='节拍均匀', timbre='音色柔和', vocal='演唱轻柔',
                lyrics='结合识别歌词表达等待，识别可能有误', uncertainty='情绪为主观估计', **changes)


class Queue:
    def __init__(self, running=None, pending=None):
        self.mutex = threading.RLock()
        self.running, self.pending = running or [], pending or []
        self.flags = {}

    def get_current_queue(self):
        return self.running, self.pending

    def set_flag(self, name, value):
        self.flags[name] = value

    def get_flags(self, reset=True):
        return {}


class Worker:
    def __init__(self, owner, args, **kwargs):
        self.owner = owner
        self.args = args
        self.terminated = False
        self.returncode = None
        owner.launched.set()

    def wait(self):
        self.owner.release.wait(5)
        if self.terminated:
            self.returncode = -15
            return self.returncode
        request = json.loads(Path(self.args[self.args.index('--request')+1]).read_text(encoding='utf-8'))
        response = Path(self.args[self.args.index('--response')+1])
        self.owner.requests.append(request)
        value = dict(schema_version=1, model_revision=emotion.MODEL_REVISION,
                     segments=[dict(index=item['index'], analysis=analysis(), raw_text='original model output')
                               for item in request['segments']])
        if self.owner.transform:
            value = self.owner.transform(value)
        if value is not None:
            response.write_text(json.dumps(value), encoding='utf-8')
        self.returncode = self.owner.returncode
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.owner.release.set()


class EmotionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name).resolve()
        self.root = self.directory/'projects'
        self.model = self.directory/'model'
        self.model.mkdir()
        (self.model/'config.json').write_text('{}', encoding='utf-8')
        (self.model/'model.safetensors').write_bytes(b'fixture weights')
        (self.model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'weight':'model.safetensors'}}), encoding='utf-8')
        for name in ('preprocessor_config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                     'special_tokens_map.json', 'chat_template.jinja', 'configuration_moss_music.py',
                     'modeling_moss_music.py', 'processing_moss_music.py'):
            (self.model/name).write_text('{}', encoding='utf-8')
        self.plan = sample_plan()
        self.project = core.project_path(self.root, self.plan['id'])
        (self.project/'audio').mkdir(parents=True)
        sf.write(self.project/'audio'/'source.wav', np.linspace(-.4, .4, self.plan['samples']), self.plan['sample_rate'])
        self.plan['segments'][0].update(final_prompt='manual H3 must survive', expanded_prompt='manually edited expansion',
            expanded_source='edited', job={'status':'completed', 'video':'old.mp4'})
        self.plan['final_video'] = 'old-final.mp4'
        core.write_plan(self.root, self.plan)
        self.queue = Queue()
        self.server = types.SimpleNamespace(prompt_queue=self.queue)
        self.launched, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.transform, self.returncode, self.requests = None, 0, []
        for context in (patch.object(emotion, 'MODEL_DIRECTORY', self.model),
                        patch.object(emotion, 'readiness', return_value=(True, '')),
                        patch.object(emotion, '_prepare_gpu', return_value=True),
                        patch.object(emotion.subprocess, 'Popen', side_effect=lambda *a, **kw: Worker(self, *a, **kw))):
            context.start()
            self.addCleanup(context.stop)
        self.addCleanup(self.finish)

    def finish(self):
        self.release.set()
        deadline = time.monotonic()+6
        while emotion.active() and time.monotonic() < deadline:
            time.sleep(.01)

    def wait(self):
        self.finish()
        self.assertFalse(emotion.active())
        return emotion.status(self.root, self.plan['id'])['job']

    def start(self, indices=None, force=False):
        plan = core.read_plan(self.root, self.plan['id'])
        return emotion.start(self.root, self.plan['id'], plan['revision'], indices or [0], self.server, force)

    def test_real_source_range_and_complete_provenance(self):
        self.start([1])
        self.assertEqual(self.wait()['status'], 'completed')
        request = self.requests[0]
        self.assertEqual(Path(request['audio_path']), self.project/'audio'/'source.wav')
        self.assertEqual(request['segments'], [dict(index=1, start_sample=1000, end_sample=2000,
                                                   sample_rate=100, lyrics=self.plan['segments'][1]['text'])])
        saved = core.read_plan(self.root, self.plan['id'])['segments'][1]['music_emotion']
        self.assertEqual(saved['input']['model_revision'], emotion.MODEL_REVISION)
        self.assertEqual(len(saved['input']['source_sha256']), 64)
        self.assertEqual(saved['raw_text'], 'original model output')

    def test_success_preserves_manual_content_and_old_video_only_marks_changed_row(self):
        self.plan['segments'][0]['prompt'] += '\n用户自己写的表情。'
        core.write_plan(self.root, self.plan)
        self.start()
        self.assertEqual(self.wait()['status'], 'completed')
        saved = core.read_plan(self.root, self.plan['id'])
        row = saved['segments'][0]
        self.assertIn('用户自己写的表情。', row['prompt'])
        self.assertIn('2.5/5（模型估计）', row['prompt'])
        self.assertEqual(row['final_prompt'], 'manual H3 must survive')
        self.assertEqual(row['expanded_prompt'], 'manually edited expansion')
        self.assertEqual(row['job']['video'], 'old.mp4')
        self.assertTrue(row['needs_regeneration'])
        self.assertEqual(saved['changed_segments'], [0])
        self.assertEqual(saved['revision'], 2)
        self.assertFalse(saved['approved'])
        self.assertTrue(saved['final_stale'])
        self.assertNotIn('needs_regeneration', saved['segments'][1])

    def test_cache_hit_keeps_revision_and_does_not_launch_worker(self):
        self.start()
        self.wait()
        self.start()
        job = self.wait()
        self.assertEqual(job['items'][0]['status'], 'cached')
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(core.read_plan(self.root, self.plan['id'])['revision'], 2)

    def test_force_replaces_owned_block_without_duplicates(self):
        self.start()
        self.wait()
        self.start(force=True)
        self.wait()
        row = core.read_plan(self.root, self.plan['id'])['segments'][0]
        self.assertEqual(row['prompt'].count('【音乐情绪分析】'), 1)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(core.read_plan(self.root, self.plan['id'])['revision'], 2)

    def test_cache_binding_changes_with_audio_lyrics_range_model_or_rules(self):
        source = emotion._source(self.project, self.plan)
        row = self.plan['segments'][0]
        key = emotion.segment_input(source, row, 'model-a')[1]
        for field, value in [('start_sample', 1), ('end_sample', 999), ('text', '新的歌词')]:
            changed = dict(row, **{field:value})
            self.assertNotEqual(key, emotion.segment_input(source, changed, 'model-a')[1])
        self.assertNotEqual(key, emotion.segment_input(dict(source, sha256='different'), row, 'model-a')[1])
        self.assertNotEqual(key, emotion.segment_input(source, row, 'model-b')[1])
        with patch.object(emotion, 'RULE_VERSION', emotion.RULE_VERSION + 1):
            self.assertNotEqual(key, emotion.segment_input(source, row, 'model-a')[1])

    def test_source_hash_cached_and_stat_change_is_detected(self):
        first = emotion._source(self.project, self.plan)
        with patch.object(Path, 'open', side_effect=AssertionError('Repeated hash read')):
            second = emotion._source(self.project, self.plan)
        self.assertEqual(first['sha256'], second['sha256'])
        sf.write(self.project/'audio'/'source.wav', np.zeros(self.plan['samples']), 100)
        self.assertNotEqual(first['sha256'], emotion._source(self.project, self.plan)['sha256'])

    def test_range_edit_invalidates_and_removes_annotation(self):
        self.start([0, 1])
        self.wait()
        plan = core.read_plan(self.root, self.plan['id'])
        submitted = copy.deepcopy(plan['segments'])
        submitted[0]['end'] = 9.5
        core.edit_plan(plan, submitted, self.project)
        for index in [0, 1]:
            self.assertEqual(plan['segments'][index]['music_emotion']['status'], 'stale')
            self.assertNotIn('【音乐情绪分析】', plan['segments'][index]['prompt'])

    def test_lyrics_change_invalidates_without_erasing_other_brief(self):
        self.start()
        self.wait()
        plan = core.read_plan(self.root, self.plan['id'])
        row = plan['segments'][0]
        row['prompt'] += '\n手工附言。'
        row['text'] = '新歌词'
        self.assertEqual(emotion.invalidate_changed(plan), [0])
        self.assertIn('手工附言。', row['prompt'])
        self.assertNotIn('【音乐情绪分析】', row['prompt'])

    def test_other_project_cannot_start_until_worker_exits_and_cancel_is_scoped(self):
        self.release.clear()
        self.start()
        self.assertTrue(self.launched.wait(1))
        other = dict(self.plan, id='b'*32)
        core.write_plan(self.root, other)
        with self.assertRaisesRegex(ValueError, '正在分析'):
            emotion.start(self.root, other['id'], 1, [0], self.server)
        emotion.cancel(self.root, other['id'])
        self.assertTrue(emotion.active(self.plan['id']))
        with self.assertRaises(ValueError):
            emotion.require_inactive()
        emotion.require_inactive(other['id'])
        emotion.cancel(self.root, self.plan['id'])
        job = self.wait()
        self.assertEqual(job['status'], 'cancelled')
        self.assertEqual(job['done'], 0)
        self.assertEqual(core.read_plan(self.root, self.plan['id'])['revision'], 1)

    def test_missing_dependency_and_model_return_no_fake_analysis(self):
        with patch.object(emotion, 'readiness', return_value=(False, '缺少本地模型')):
            with self.assertRaisesRegex(ValueError, '缺少本地模型'):
                self.start()
        self.assertIsNone(emotion.status(self.root, self.plan['id'])['job'])
        self.assertNotIn('music_emotion', core.read_plan(self.root, self.plan['id'])['segments'][0])

    def test_worker_failure_does_not_create_analysis(self):
        self.returncode = 1
        self.transform = lambda value: None
        self.start()
        self.assertEqual(self.wait()['status'], 'failed')
        self.assertEqual(core.read_plan(self.root, self.plan['id'])['revision'], 1)

    def test_partial_failure_applies_only_returned_valid_segments(self):
        self.returncode = 1
        self.transform = lambda value: dict(value, segments=value['segments'][:1], errors=[{'index':1,'error':'OOM'}])
        self.start([0, 1])
        self.assertEqual(self.wait()['status'], 'failed')
        saved = core.read_plan(self.root, self.plan['id'])
        self.assertEqual(saved['changed_segments'], [0])
        self.assertNotIn('music_emotion', saved['segments'][1])

    def test_forged_response_index_or_revision_cannot_write_plan(self):
        for transform in (lambda value: dict(value, model_revision='old'),
                          lambda value: dict(value, segments=[dict(value['segments'][0], index=2)])):
            self.transform = transform
            self.start()
            self.assertEqual(self.wait()['status'], 'failed')
            self.assertEqual(core.read_plan(self.root, self.plan['id'])['revision'], 1)

    def test_external_edit_during_inference_retains_stale_candidate(self):
        self.release.clear()
        self.start()
        self.assertTrue(self.launched.wait(1))
        plan = core.read_plan(self.root, self.plan['id'])
        plan['revision'] += 1
        plan['segments'][0]['prompt'] = '外部用户修改'
        core.write_plan(self.root, plan)
        self.release.set()
        job = self.wait()
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['items'][0]['apply_status'], 'stale')
        self.assertEqual(core.read_plan(self.root, self.plan['id'])['segments'][0]['prompt'], '外部用户修改')

    def test_unversioned_lyrics_or_prompt_edit_rejected_on_commit(self):
        self.release.clear()
        self.start()
        self.assertTrue(self.launched.wait(1))
        plan = core.read_plan(self.root, self.plan['id'])
        plan['segments'][0]['text'] = 'externally changed'
        core.write_plan(self.root, plan)
        self.release.set()
        self.assertEqual(self.wait()['items'][0]['apply_status'], 'stale')
        self.assertNotIn('music_emotion', core.read_plan(self.root, self.plan['id'])['segments'][0])

    def test_other_h3_controller_blocks_analysis_between_queued_segments(self):
        controller = importlib.import_module('h3lv_test.controller')
        with patch.dict(controller.TASKS, {'b'*32: object()}):
            with self.assertRaisesRegex(ValueError, 'H3 顺序生成任务'):
                self.start()
        self.assertFalse(self.launched.is_set())
        self.assertFalse(emotion.active())
        self.assertEqual(self.queue.flags, {})

    def test_busy_comfy_queue_refuses_without_setting_unload_flag(self):
        self.queue.running = ['unrelated task']
        with self.assertRaisesRegex(ValueError, 'ComfyUI'):
            self.start()
        self.assertEqual(self.queue.flags, {})
        self.assertFalse(emotion.active())

    def test_invalid_selection_revision_and_force_are_rejected(self):
        for revision, indices, force in [(True, [0], False), (2, [0], False), (1, [True], False),
                                         (1, [0,0], False), (1, [-1], False), (1, [0], 'yes'), (1, [{}], False)]:
            with self.assertRaises(ValueError):
                emotion.start(self.root, self.plan['id'], revision, indices, self.server, force)

    def test_model_signature_requires_all_indexed_shards_and_processor_files(self):
        signature = emotion.model_signature()
        index = self.model/'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map':{'a':'model.safetensors', 'b':'second.safetensors'}}), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'second.safetensors'):
            emotion.model_signature()
        (self.model/'second.safetensors').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'second.safetensors'):
            emotion.model_signature()
        (self.model/'second.safetensors').write_bytes(b'second')
        self.assertNotEqual(signature, emotion.model_signature())
        (self.model/'preprocessor_config.json').unlink()
        with self.assertRaisesRegex(ValueError, 'preprocessor_config.json'):
            emotion.model_signature()

    def test_native_unload_flag_only_set_while_queue_idle(self):
        self.queue.running = ['other job']
        with self.assertRaises(ValueError):
            prepare_gpu(self.server, threading.Event())
        self.assertEqual(self.queue.flags, {})
        self.queue.running = []
        with patch.object(threading.Event, 'wait', return_value=False):
            self.assertTrue(prepare_gpu(self.server, threading.Event()))
        self.assertEqual(self.queue.flags, {'unload_models':True})

    def test_native_unload_does_not_start_worker_if_new_task_arrives(self):
        def arrive(timeout):
            self.queue.running = ['arriving job']
            return False
        with patch.object(threading.Event, 'wait', side_effect=arrive), self.assertRaises(ValueError):
            prepare_gpu(self.server, threading.Event())
        self.assertFalse(self.launched.is_set())

    def test_service_restart_status_is_read_only_failure_not_fake_completion(self):
        base = emotion._directory(self.root, self.plan['id'])
        base.mkdir(parents=True)
        file = base/'latest.json'
        file.write_text(json.dumps(dict(id='abandoned', status='running', done=1, total=3, index=0)), encoding='utf-8')
        before = file.read_bytes()
        value = emotion.status(self.root, self.plan['id'])
        self.assertEqual(value['job']['status'], 'failed')
        self.assertEqual(file.read_bytes(), before)

    def test_model_labels_and_fields_cannot_inject_owned_block_markers(self):
        for marker in (emotion.BLOCK_START, emotion.BLOCK_END):
            for field, value in [('labels', ['温柔', marker]), ('summary', marker)]:
                result = dict(analysis(), **{field:value})
                with self.assertRaisesRegex(ValueError, '保留标记'):
                    emotion.annotate_brief('用户正文', result)

    def test_visual_type_brief_preview_keeps_valid_emotion_and_leaves_plan_unchanged(self):
        self.start()
        self.wait()
        plan = core.read_plan(self.root, self.plan['id'])
        before = copy.deepcopy(plan)
        preview = core.preview_segment_brief(plan, 0, 'environment')
        self.assertIn('【音乐情绪分析】', preview)
        self.assertEqual(plan, before)


if __name__ == '__main__':
    unittest.main()
