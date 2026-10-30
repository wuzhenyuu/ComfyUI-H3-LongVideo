import hashlib
import importlib
import io
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType('h3lv_test')
package.__path__ = [str(ROOT)]
sys.modules.setdefault('h3lv_test', package)
videos = importlib.import_module('h3lv_test.video_materials')


def make_video(target, duration=2, fps=24, audio=False, color='red'):
    codec = 'libvpx-vp9' if target.suffix == '.webm' else 'libx264'
    arguments = ['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
                 f'color=c={color}:s=64x48:r={fps}:d={duration}']
    if audio:
        arguments += ['-f', 'lavfi', '-i', f'sine=frequency=440:duration={duration + 2}']
    arguments += ['-map', '0:v:0']
    if audio:
        arguments += ['-map', '1:a:0', '-c:a', 'libopus' if target.suffix == '.webm' else 'aac']
    arguments += ['-c:v', codec, '-pix_fmt', 'yuv420p', '-y', str(target)]
    subprocess.run(arguments, check=True, capture_output=True)
    return target


class VideoMaterialsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
            raise unittest.SkipTest('需要既有 FFmpeg / FFprobe 运行实媒体测试。')
        cls.fixture_directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture_directory.cleanup)
        cls.fixtures = Path(cls.fixture_directory.name).resolve()
        cls.two = make_video(cls.fixtures/'two.mp4')
        cls.fifteen = make_video(cls.fixtures/'fifteen.mp4', 15)
        cls.short = make_video(cls.fixtures/'short.mp4', 47 / 24)
        cls.long = make_video(cls.fixtures/'long.mp4', 361 / 24)
        cls.eight = make_video(cls.fixtures/'eight.mp4', 8)
        cls.audio = make_video(cls.fixtures/'audio.mp4', audio=True, fps=12)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.project = Path(temporary.name).resolve()

    def store(self, source=None):
        source = source or self.two
        return videos.store_reference(self.project, source.name, source.read_bytes())

    def test_probe_uses_video_stream_duration_when_audio_is_longer(self):
        info = videos.probe(self.audio)
        self.assertEqual(info['duration'], 2.0)
        self.assertTrue(info['has_audio'])
        self.assertEqual((info['width'], info['height'], info['fps']), (64, 48, 12.0))

    def test_exact_duration_boundaries_and_combined_limit(self):
        self.assertEqual(videos.validate_group([self.two])[0]['duration'], 2)
        self.assertEqual(videos.validate_group([self.fifteen])[0]['duration'], 15)
        for source in (self.short, self.long):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, '2–15'):
                videos.validate_group([source])
        with self.assertRaisesRegex(ValueError, '总时长'):
            videos.validate_group([self.eight, self.eight])
        with self.assertRaisesRegex(ValueError, '最多 3'):
            videos.validate_group([self.two] * 4)
        self.assertEqual(len(videos.validate_group([self.two] * 3)), 3)

    def test_nearby_outside_boundaries_are_not_accepted_as_frame_tolerance(self):
        for duration in (1.99999, 15.00001):
            with self.subTest(duration=duration), patch.object(videos, 'probe', return_value={'duration': duration}):
                with self.assertRaisesRegex(ValueError, '2–15'):
                    videos.validate_group([self.two])

    def test_supported_containers_are_decoded_and_not_misread_by_quantized_timestamps(self):
        for suffix in ('.mov', '.mkv', '.webm', '.avi'):
            with self.subTest(suffix=suffix):
                source = make_video(self.project/f'video{suffix}')
                self.assertEqual(videos.probe(source)['duration'], 2)
                self.assertEqual(tuple(videos.frame_tensors({'video_paths': [str(source)]})[0].shape),
                                 (48, 48, 64, 3))

    def test_fractional_cfr_and_variable_rate_use_one_normalized_frame_contract(self):
        for fps, duration in [('30', 2.1), ('24000/1001', 2.002)]:
            with self.subTest(fps=fps):
                source = make_video(self.project/f'fps_{fps.replace("/", "_")}.mp4', duration, fps)
                info = videos.probe(source)
                result = videos.frame_tensors({'video_paths': [str(source)]})[0]
                expected = math.floor(info['duration'] * 24 + 1e-7)
                self.assertEqual(result.shape[0], expected)
                self.assertLessEqual(abs(result.shape[0] / 24 - info['duration']), 1 / 24)
                self.assertLessEqual(result.shape[0], 360)
                self.assertEqual(videos.sample_timestamps(info['duration'])[-1], (expected - 1) / 24)
                self.assertTrue(videos.contact_sheet(source, self.project/'cache').is_file())
        source = self.project/'vfr.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                        'testsrc2=s=64x48:r=30:d=2.1', '-vf', 'select=not(eq(mod(n\\,5)\\,1))',
                        '-fps_mode', 'vfr', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', str(source)],
                       check=True, capture_output=True)
        info = videos.probe(source)
        self.assertAlmostEqual(info['duration'], 2.1)
        self.assertEqual(videos.frame_tensors({'video_paths': [str(source)]})[0].shape[0], 50)
        self.assertTrue(videos.contact_sheet(source, self.project/'cache').is_file())

    def test_fractional_group_normalization_cannot_exceed_fifteen_seconds(self):
        sources = [make_video(self.project/f'group_{index}.mp4', duration, 30)
                   for index, duration in enumerate((2.1, 2.1, 10.8))]
        self.assertEqual(sum(row['duration'] for row in videos.validate_group(sources)), 15)
        tensors = videos.frame_tensors({'video_paths': [str(source) for source in sources]})
        self.assertEqual([tensor.shape[0] for tensor in tensors], [50, 50, 259])
        self.assertLessEqual(sum(tensor.shape[0] for tensor in tensors), 360)

    def test_default_inheritance_custom_override_empty_and_ordered_packet(self):
        first, second = self.store(), self.store(self.audio)
        plan = {'default_video_refs': [first['name'], second['name']], 'default_video_note': ' 运镜 '}
        row = {'video_reference_source': 'default', 'video_refs': [second['name']], 'video_note': '本段'}
        packet = videos.packet(plan, row, self.project)
        self.assertEqual(packet['video_refs'], plan['default_video_refs'])
        self.assertEqual(packet['video_note'], '运镜')
        self.assertEqual(packet['video_reference_source'], 'default')
        self.assertEqual(packet['video_durations'], [2.0, 2.0])
        self.assertEqual(packet['video_hashes'][0], hashlib.sha256(self.two.read_bytes()).hexdigest())
        self.assertEqual([Path(p).name for p in packet['video_paths']], packet['video_refs'])
        row['video_reference_source'] = 'custom'
        self.assertEqual(videos.effective(plan, row, self.project)['refs'], [second['name']])
        row['video_refs'] = []
        self.assertEqual(videos.packet(plan, row, self.project)['video_refs'], [])
        self.assertEqual(videos.effective({}, {}), {'refs': [], 'note': '', 'source': 'default'})
        with self.assertRaisesRegex(ValueError, '来源'):
            videos.effective(plan, {'video_reference_source': 'outside'})

    def test_invalid_names_and_count_are_rejected_without_writing(self):
        for value in ('../test.mp4', 'folder\\test.mp4', 'C:test.mp4', 'https://host/test.mp4', 'image.png', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                videos.path(self.project, value)
        with self.assertRaises(ValueError):
            videos.normalize_names('video.mp4')
        with self.assertRaisesRegex(ValueError, '重复'):
            videos.normalize_names(['video.mp4'] * 2)
        with self.assertRaisesRegex(ValueError, '最多 3'):
            videos.normalize_names(['video.mp4'] * 4)
        self.assertEqual(list(self.project.iterdir()), [])

    def test_store_preserves_original_and_only_validates_selected_group_total(self):
        first = self.store(self.eight)
        second = self.store(self.eight)
        self.assertEqual(videos.path(self.project, first['name']).read_bytes(), self.eight.read_bytes())
        self.assertEqual(videos.metadata(self.project, first['name'])['duration'], 8)
        with self.assertRaisesRegex(ValueError, '总时长'):
            videos.normalize_names([first['name'], second['name']], self.project)

    def test_fake_extension_audio_only_truncation_and_network_protocol_are_rejected(self):
        source = self.project/'fake.mp4'
        picture = io.BytesIO()
        Image.new('RGB', (10, 10)).save(picture, format='PNG')
        for content in (b'not a video', picture.getvalue(), self.two.read_bytes()[:1000]):
            source.write_bytes(content)
            with self.subTest(size=len(content)), self.assertRaises(ValueError):
                videos.probe(source)
        source = self.project/'audio_only.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=duration=2',
                        '-c:a', 'aac', '-y', str(source)], check=True, capture_output=True)
        with self.assertRaises(ValueError):
            videos.probe(source)
        with patch.object(videos, '_run') as run, self.assertRaises(ValueError):
            videos.probe('https://example.com/remote.mp4')
        run.assert_not_called()

    def test_truncated_valid_length_containers_are_not_accepted_as_shorter_videos(self):
        for suffix in ('.mp4', '.mkv', '.webm', '.avi'):
            with self.subTest(suffix=suffix):
                source = make_video(self.project/f'original{suffix}', 8)
                content = source.read_bytes()
                for fraction in (0.5, 0.95):
                    truncated = self.project/f'truncated_{fraction}{suffix}'
                    truncated.write_bytes(content[:int(len(content) * fraction)])
                    with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                        videos.validate_group([truncated])
        renamed = self.project/'renamed.mkv'
        renamed.write_bytes(self.two.read_bytes())
        with self.assertRaisesRegex(ValueError, '扩展名'):
            videos.probe(renamed)

    def test_rejected_upload_does_not_create_video_references(self):
        for filename, content in [('fake.mp4', b'bad'), ('empty.mp4', b''), ('small.mp4', self.short.read_bytes())]:
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                videos.store_reference(self.project, filename, content)
        with patch.object(videos, 'MAX_VIDEO_BYTES', 2), self.assertRaisesRegex(ValueError, '512 MiB'):
            self.store()
        self.assertFalse((self.project/'video_refs').exists())

    def test_frame_tensors_are_24fps_rgb_cpu_float_and_do_not_expose_audio(self):
        import numpy as np
        import torch
        with patch.object(np, 'stack', side_effect=AssertionError('must not stack full videos')):
            result = videos.frame_tensors({'video_paths': [str(self.audio)]})
        self.assertEqual(len(result), 3)
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])
        tensor = result[0]
        self.assertEqual(tuple(tensor.shape), (48, 48, 64, 3))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device.type, 'cpu')
        self.assertGreater(float(tensor[0, 0, 0, 0]), 0.9)
        self.assertTrue(bool(torch.all((tensor >= 0) & (tensor <= 1))))
        self.assertEqual(videos.frame_tensors({'video_paths': []}), (None, None, None))

    def test_size_limits_are_applied_without_upscaling(self):
        for width, height in [(3840, 2160), (2160, 3840), (8000, 300), (64, 48)]:
            actual_width, actual_height = videos._size(width, height)
            self.assertLessEqual(actual_width, width)
            self.assertLessEqual(actual_height, height)
            self.assertLessEqual(min(actual_width, actual_height), 768)
            self.assertLessEqual(actual_width * actual_height, 768 * 1344)
            self.assertLess(abs(actual_width / actual_height - width / height), 0.1)
        self.assertEqual(videos._size(64, 48), (64, 48))

    def test_failed_decode_reaps_ffmpeg_child(self):
        children = []
        real_popen = videos.subprocess.Popen

        def record(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child

        videos.probe(self.two)
        with patch.object(videos.subprocess, 'Popen', side_effect=record), \
                patch.object(videos, '_frame_filter', return_value='no_such_video_filter'):
            with self.assertRaises(ValueError):
                videos.frame_tensors({'video_paths': [str(self.two)]})
        self.assertTrue(children)
        self.assertTrue(all(child.poll() is not None and child.stdout.closed for child in children))

    def test_thumbnail_contact_sheet_and_cache_follow_file_identity(self):
        source = self.project/'video.mp4'
        shutil.copyfile(self.two, source)
        thumb = videos.thumbnail(source)
        with Image.open(io.BytesIO(thumb)) as image:
            self.assertEqual(image.format, 'PNG')
            self.assertEqual(image.size, (64, 48))
        cache = self.project/'cache'
        first = videos.contact_sheet(source, cache)
        with Image.open(first) as sheet:
            self.assertEqual(sheet.size, (1280, 440))
        with patch.object(videos, '_sample_images', side_effect=AssertionError('cache should be reused')):
            self.assertEqual(videos.contact_sheet(source, cache), first)
        shutil.copyfile(self.eight, source)
        self.assertEqual(videos.probe(source)['duration'], 8)
        self.assertNotEqual(videos.contact_sheet(source, cache), first)

    def test_sample_timestamps_match_normalized_indices_and_remain_within_duration(self):
        for duration in (2, 7.5, 15):
            times = videos.sample_timestamps(duration)
            self.assertEqual(len(times), 8)
            self.assertEqual(times[0], 0)
            self.assertEqual(times[-1], (int(duration * 24) - 1) / 24)
            self.assertTrue(all(timestamp <= duration for timestamp in times))
            self.assertEqual(times, sorted(times))
            self.assertTrue(all(abs(timestamp * 24 - round(timestamp * 24)) < 1e-7 for timestamp in times))

    def test_project_directory_and_file_symlinks_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            outside = Path(temporary).resolve()
            link = self.project/'video_refs'
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                self.skipTest(f'当前账户不能创建符号链接：{error}')
            self.addCleanup(link.unlink, missing_ok=True)
            with self.assertRaisesRegex(ValueError, '不属于当前项目'):
                self.store()
            self.assertEqual(list(outside.iterdir()), [])
            link.unlink()
            link.mkdir()
            outside_file = outside/'video.mp4'
            shutil.copyfile(self.two, outside_file)
            file_link = link/'video.mp4'
            file_link.symlink_to(outside_file)
            with self.assertRaisesRegex(ValueError, '不属于当前项目'):
                videos.metadata(self.project, 'video.mp4')
            file_link.unlink()
            link.rmdir()


if __name__ == '__main__':
    unittest.main()
