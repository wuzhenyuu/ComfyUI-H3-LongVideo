"""Local, silent video references for motion guidance and prompt expansion."""
from collections import OrderedDict
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import uuid


MAX_VIDEO_BYTES = 512 * 1024 * 1024
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.webm', '.mkv', '.avi'}
MAX_VIDEO_REFERENCES = 3
VIDEO_FPS = 24
MIN_VIDEO_SECONDS = 2
MAX_VIDEO_SECONDS = 15
MAX_VIDEO_FRAMES = MAX_VIDEO_SECONDS * VIDEO_FPS
_EPSILON = 1e-7
_PROBE_CACHE = OrderedDict()
_CACHE_LOCK = threading.RLock()
_FORMATS = {'mov', 'mp4', 'matroska', 'webm', 'avi'}


def _inside(root, target):
    root, target = Path(root).resolve(), Path(target).resolve()
    if not target.is_relative_to(root):
        raise ValueError('文件不属于当前项目。')
    return target


def reference_directory(project):
    return _inside(project, Path(project)/'video_refs')


def _filename(name):
    if (not isinstance(name, str) or not name.strip() or Path(name).name != name
            or any(char in name for char in '<>:"/\\|?*\0')):
        raise ValueError('参考视频文件名无效。')
    if Path(name).suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError('参考视频只支持 MP4、MOV、WEBM、MKV 或 AVI 格式。')
    return name


def path(project, name):
    root = reference_directory(project)
    return _inside(root, root/_filename(name))


def _local_file(value):
    # Resolve a local Path, never pass URL-like strings or playlist protocols to FFmpeg.
    candidate = Path(value)
    _filename(candidate.name)
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise ValueError(f'参考视频已丢失：{candidate.name}。请重新上传。')
    if not 0 < candidate.stat().st_size <= MAX_VIDEO_BYTES:
        raise ValueError('参考视频内容为空，或单个视频超过 512 MiB。')
    return candidate


def _identity(source):
    stat = source.stat()
    return (str(source), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


def _executable(name):
    result = shutil.which(name)
    if not result:
        raise ValueError(f'未找到 {name}，请配置现有 FFmpeg 到 PATH。插件不会自动安装。')
    return result


def _run(arguments, timeout=120):
    try:
        result = subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout, creationflags=(subprocess.CREATE_NO_WINDOW
                                                               if os.name == 'nt' else 0))
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError('参考视频读取失败或超时。') from error
    if result.returncode or result.stderr.strip():
        raise ValueError('上传内容不是有效、完整的参考视频。')
    return result.stdout


def _number(value, fallback=0):
    try:
        result = float(Fraction(str(value)))
        return result if math.isfinite(result) else fallback
    except (ValueError, ZeroDivisionError, TypeError):
        return fallback


def _container_complete(source, formats):
    """Check declared box/chunk lengths; demuxers may silently accept an early EOF."""
    length = source.stat().st_size
    with source.open('rb') as handle:
        if 'mov' in formats or 'mp4' in formats:
            while handle.tell() < length:
                start = handle.tell()
                header = handle.read(8)
                if len(header) != 8:
                    raise ValueError('参考视频文件被截断。')
                size = int.from_bytes(header[:4], 'big')
                minimum = 8
                if size == 1:
                    extended = handle.read(8)
                    if len(extended) != 8:
                        raise ValueError('参考视频文件被截断。')
                    size, minimum = int.from_bytes(extended, 'big'), 16
                if size == 0:
                    size = length - start
                if size < minimum or start + size > length:
                    raise ValueError('参考视频文件被截断。')
                handle.seek(start + size)
        elif 'avi' in formats:
            while handle.tell() < length:
                start = handle.tell()
                header = handle.read(12)
                if len(header) != 12 or header[:4] != b'RIFF':
                    raise ValueError('参考视频文件被截断。')
                size = int.from_bytes(header[4:8], 'little') + 8
                if size < 12 or start + size > length:
                    raise ValueError('参考视频文件被截断。')
                handle.seek(start + size + (size % 2))
        elif 'matroska' in formats or 'webm' in formats:
            def vint(identifier=False):
                first = handle.read(1)
                if not first or not first[0]:
                    raise ValueError('参考视频文件被截断。')
                size = 9 - first[0].bit_length()
                if size > (4 if identifier else 8):
                    raise ValueError('参考视频容器数据无效。')
                tail = handle.read(size - 1)
                if len(tail) != size - 1:
                    raise ValueError('参考视频文件被截断。')
                raw = int.from_bytes(first + tail, 'big')
                value = raw if identifier else raw & ((1 << (7 * size)) - 1)
                return value, (not identifier and value == (1 << (7 * size)) - 1)

            while handle.tell() < length:
                identifier, _ = vint(True)
                size, unknown = vint()
                if unknown:
                    # An unknown-length Segment/Cluster is used by streaming writers.
                    # Continue checking its children rather than skipping to EOF.
                    if identifier not in (0x18538067, 0x1F43B675):
                        raise ValueError('参考视频容器数据无效。')
                else:
                    end = handle.tell() + size
                    if end > length:
                        raise ValueError('参考视频文件被截断。')
                    handle.seek(end)


def _decoded_duration(source, stream_index, expected_frames, avi=False, declared=0):
    """Read one native frame at a time, including the final packet, without RGB copies."""
    import av
    first = previous = last_end = None
    step = None
    count = 0
    constant_rate = True
    try:
        with av.open(str(source), options={'protocol_whitelist': 'file,pipe',
                                          'err_detect': 'explode'}) as container:
            stream = container.streams[stream_index]
            rate = stream.average_rate or stream.base_rate
            for frame in container.decode(stream):
                if frame.is_corrupt or frame.pts is None or frame.time_base is None:
                    raise ValueError('参考视频包含损坏或缺少时间戳的画面。')
                timestamp = Fraction(frame.pts) * frame.time_base
                if previous is not None:
                    if timestamp <= previous and not avi:
                        raise ValueError('参考视频的画面时间轴无效。')
                    step = timestamp - previous
                first = timestamp if first is None else first
                # Matroska/WebM timestamps are often quantized to milliseconds. Prove
                # a constant cadence from all decoded frames, then use its exact rate
                # instead of treating a rounded last packet duration as real trimming.
                if not rate or abs(timestamp - (first + Fraction(count, 1) / rate)) > frame.time_base / 2:
                    constant_rate = False
                duration = (Fraction(frame.duration) * frame.time_base
                            if frame.duration and frame.duration > 0 else step)
                if not duration:
                    duration = Fraction(1, 1) / rate if rate and rate > 0 else None
                if not duration:
                    raise ValueError('无法确认参考视频的真实画面时长。')
                last_end = timestamp + duration
                previous = timestamp
                count += 1
                # A misleading short container header must not cause an unbounded scan.
                if float(last_end - first) > MAX_VIDEO_SECONDS + 1 + _EPSILON:
                    raise ValueError('每个参考视频必须在 2–15 秒之间。')
        if not count or (expected_frames and count != expected_frames):
            raise ValueError('参考视频不完整，实际画面数量与视频记录不符。')
    except (av.error.FFmpegError, OSError, OverflowError) as error:
        raise ValueError('上传内容不是有效、完整的参考视频。') from error
    if rate and (constant_rate or avi):
        duration = float(Fraction(count, 1) / rate)
        last_start = float(Fraction(count - 1, 1) / rate)
    else:
        duration, last_start = float(last_end - first), float(previous - first)
    if declared:
        # MOV/MP4 edit lists can end inside the last presented frame (especially
        # VFR with B-frames). The stream duration is its real playback endpoint;
        # verify that decoded frames actually reach it, without adding a frame's
        # tolerance to the permitted 2–15 second source duration.
        if not last_start - _EPSILON < declared <= duration + _EPSILON:
            raise ValueError('参考视频不完整，真实画面时长与视频记录不符。')
        duration = declared
    return duration, count


def probe(value):
    """Return verified video-stream metadata; audio/container duration is never used."""
    source = _local_file(value)
    identity = _identity(source)
    with _CACHE_LOCK:
        if identity in _PROBE_CACHE:
            _PROBE_CACHE.move_to_end(identity)
            return dict(_PROBE_CACHE[identity])
    raw = _run([_executable('ffprobe'), '-v', 'error', '-protocol_whitelist', 'file,pipe',
                '-format_whitelist', 'mov,matroska,webm,avi', '-show_streams', '-show_format',
                '-of', 'json', str(source)])
    try:
        data = json.loads(raw)
        formats = set(data['format']['format_name'].split(','))
        streams = data['streams']
        stream = next(row for row in streams if row.get('codec_type') == 'video'
                      and not row.get('disposition', {}).get('attached_pic'))
        if not formats.intersection(_FORMATS):
            raise ValueError('参考视频实际格式不受支持。')
        expected = {'.mp4': {'mov', 'mp4'}, '.mov': {'mov', 'mp4'},
                    '.webm': {'matroska', 'webm'}, '.mkv': {'matroska', 'webm'}, '.avi': {'avi'}}
        if not formats.intersection(expected[source.suffix.lower()]):
            raise ValueError('参考视频扩展名与实际容器格式不一致。')
        _container_complete(source, formats)
        width, height = int(stream['width']), int(stream['height'])
        if width <= 0 or height <= 0:
            raise ValueError('参考视频尺寸无效。')
        declared = (_number(stream.get('duration_ts')) * _number(stream.get('time_base'))
                    or _number(stream.get('duration')))
        if declared > MAX_VIDEO_SECONDS + _EPSILON:
            raise ValueError('每个参考视频必须在 2–15 秒之间。')
        count = int(stream.get('nb_frames', 0)) if str(stream.get('nb_frames', '')).isdigit() else 0
        duration, count = _decoded_duration(source, stream['index'], count, formats == {'avi'}, declared)
        result = {'duration': duration, 'width': width, 'height': height,
                  'fps': _number(stream.get('avg_frame_rate')) or count / duration,
                  'has_audio': any(row.get('codec_type') == 'audio' for row in streams),
                  'format': data['format']['format_name']}
    except (KeyError, StopIteration, TypeError, ZeroDivisionError, json.JSONDecodeError) as error:
        raise ValueError('上传内容不是有效、完整的参考视频。') from error
    if identity != _identity(source):
        raise ValueError('读取过程中参考视频已改变，请重试。')
    with _CACHE_LOCK:
        _PROBE_CACHE[identity] = dict(result)
        _PROBE_CACHE.move_to_end(identity)
        while len(_PROBE_CACHE) > 128:
            _PROBE_CACHE.popitem(last=False)
    return result


def validate_group(paths):
    if not isinstance(paths, (list, tuple)) or len(paths) > MAX_VIDEO_REFERENCES:
        raise ValueError('每段最多 3 个参考视频。')
    values = [probe(source) for source in paths]
    if any(not MIN_VIDEO_SECONDS - _EPSILON <= item['duration'] <= MAX_VIDEO_SECONDS + _EPSILON
           for item in values):
        raise ValueError('每个参考视频必须在 2–15 秒之间。')
    if math.fsum(item['duration'] for item in values) > MAX_VIDEO_SECONDS + _EPSILON:
        raise ValueError('选中参考视频的总时长不能超过 15 秒。')
    return values


def normalize_names(names, directory=None):
    if names is None or names == '':
        return []
    if not isinstance(names, (list, tuple)):
        raise ValueError('参考视频必须按顺序保存为列表。')
    if len(names) > MAX_VIDEO_REFERENCES:
        raise ValueError('每段最多 3 个参考视频。')
    result = [_filename(name) for name in names]
    if len(set(result)) != len(result):
        raise ValueError('不能重复选择同一参考视频。')
    if directory is not None:
        validate_group([path(directory, name) for name in result])
    return result


def effective(plan, row, directory=None):
    source = row.get('video_reference_source', 'custom' if row.get('video_refs') else 'default')
    if source not in ('default', 'custom'):
        raise ValueError('参考视频来源无效。')
    names = plan.get('default_video_refs', []) if source == 'default' else row.get('video_refs', [])
    note = plan.get('default_video_note', '') if source == 'default' else row.get('video_note', '')
    if not isinstance(note, str):
        raise ValueError('参考视频说明必须是文字。')
    return {'refs': normalize_names(names, directory), 'note': note.strip(), 'source': source}


def _sha256(source):
    digest = hashlib.sha256()
    with source.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def packet(plan, row, directory):
    value = effective(plan, row, directory)
    paths = [path(directory, name) for name in value['refs']]
    values = validate_group(paths)
    return {'video_refs': value['refs'], 'video_note': value['note'],
            'video_reference_source': value['source'], 'video_paths': [str(p) for p in paths],
            'video_hashes': [_sha256(p) for p in paths],
            'video_durations': [item['duration'] for item in values]}


def _atomic_bytes(target, content):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('xb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def store_reference(project, filename, content):
    _filename(filename)
    if not isinstance(content, (bytes, bytearray, memoryview)) or not content:
        raise ValueError('上传的参考视频内容为空。')
    if len(content) > MAX_VIDEO_BYTES:
        raise ValueError('单个参考视频不能超过 512 MiB。')
    # Validate fully before creating a project snapshot.
    with tempfile.TemporaryDirectory(prefix='h3lv-video-') as temporary:
        incoming = Path(temporary)/filename
        incoming.write_bytes(content)
        value = validate_group([incoming])[0]
    name = f'video_{uuid.uuid4().hex}{Path(filename).suffix.lower()}'
    target = path(project, name)
    _atomic_bytes(target, content)
    return {'name': name, 'metadata': value}


def metadata(project, name):
    return probe(path(project, name))


def _size(width, height, short_side=768, area=768 * 1344):
    scale = min(1.0, short_side / min(width, height), math.sqrt(area / (width * height)))
    return max(1, math.floor(width * scale)), max(1, math.floor(height * scale))


def _frame_filter(value, width, height):
    # AVI stores a constant-rate frame sequence; codecs with reordered frames can
    # expose decode-order timestamps. Rebuild that container's presentation clock.
    clock = f"setpts=N/({value['fps']}*TB)," if value['format'] == 'avi' else ''
    return f'{clock}fps={VIDEO_FPS}:round=up,scale={width}:{height}:flags=lanczos'


def _frame_count(duration):
    # Floor each source separately so a selected <=15s group stays <=360 frames.
    return min(MAX_VIDEO_FRAMES, math.floor(duration * VIDEO_FPS + _EPSILON))


def frame_tensors(material):
    """Decode RGB only at 24fps directly into the final CPU IMAGE allocation."""
    import numpy as np
    import torch
    paths = material.get('video_paths', [])
    values = validate_group(paths)
    result = []
    for source, value in zip(paths, values):
        width, height = _size(value['width'], value['height'])
        capacity = _frame_count(value['duration'])
        tensor = torch.empty((capacity, height, width, 3), dtype=torch.float32, device='cpu')
        buffer = bytearray(width * height * 3)
        view = memoryview(buffer)
        count = 0
        child = None
        with tempfile.TemporaryFile() as errors:
            try:
                child = subprocess.Popen([_executable('ffmpeg'), '-v', 'error', '-nostdin',
                    '-xerror', '-protocol_whitelist', 'file,pipe', '-noautorotate', '-i', str(source),
                    '-map', '0:V:0', '-an', '-sn', '-dn', '-vf',
                    _frame_filter(value, width, height),
                    '-frames:v', str(capacity), '-fps_mode', 'passthrough',
                    '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], stdout=subprocess.PIPE,
                    stderr=errors, creationflags=(subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0))
                while True:
                    offset = 0
                    while offset < len(buffer):
                        length = child.stdout.readinto(view[offset:])
                        if not length:
                            break
                        offset += length
                    if not offset:
                        break
                    if offset != len(buffer) or count >= capacity:
                        raise ValueError('参考视频的 24fps 画面数量或数据长度无效。')
                    pixels = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
                    tensor[count].copy_(torch.from_numpy(pixels))
                    tensor[count].div_(255)
                    count += 1
                child.wait(timeout=30)
                errors.seek(0)
                if child.returncode or errors.read(4096).strip() or count != capacity:
                    raise ValueError('参考视频解码失败，或实际画面时长已改变。')
            finally:
                if child is not None:
                    if child.poll() is None:
                        child.kill()
                    child.wait()
                    if child.stdout is not None:
                        child.stdout.close()
                view.release()
        result.append(tensor)
    return tuple(result + [None] * (MAX_VIDEO_REFERENCES - len(result)))


def sample_timestamps(duration):
    """Eight evenly spaced frame positions on the normalized 24fps timeline."""
    duration = _number(duration)
    if not MIN_VIDEO_SECONDS - _EPSILON <= duration <= MAX_VIDEO_SECONDS + _EPSILON:
        raise ValueError('每个参考视频必须在 2–15 秒之间。')
    last = max(0, _frame_count(duration) - 1)
    return [round(index * last / 7) / VIDEO_FPS for index in range(8)]


def _sample_images(source, value, directory, times, size):
    from PIL import Image
    width, height = _size(value['width'], value['height'], min(size), size[0] * size[1])
    indices = [round(timestamp * VIDEO_FPS) for timestamp in times]
    select = '+'.join(f'eq(n\\,{index})' for index in indices)
    _run([_executable('ffmpeg'), '-v', 'error', '-nostdin', '-xerror',
          '-protocol_whitelist', 'file,pipe', '-noautorotate', '-i', str(source),
          '-map', '0:V:0', '-an', '-sn', '-dn', '-vf',
          f'{_frame_filter(value, width, height)},select={select}',
          '-fps_mode', 'vfr', '-frames:v', str(len(times)), str(Path(directory)/'sample_%02d.png')])
    images = []
    for index in range(1, len(times) + 1):
        filename = Path(directory)/f'sample_{index:02d}.png'
        if not filename.is_file():
            raise ValueError('无法读取完整的参考视频抽样画面。')
        with Image.open(filename) as image:
            image.load()
            images.append(image.convert('RGB'))
    return images


def _png(image):
    output = io.BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


def thumbnail(value):
    source = _local_file(value)
    info = validate_group([source])[0]
    with tempfile.TemporaryDirectory(prefix='h3lv-video-thumb-') as temporary:
        images = _sample_images(source, info, temporary, [0.5], (256, 256))
        return _png(images[0])


def contact_sheet(value, cache_dir):
    from PIL import Image, ImageDraw
    source = _local_file(value)
    info = validate_group([source])[0]
    identity = _identity(source)
    key = hashlib.sha256(json.dumps(['video-sheet-24fps-floor-v2', *identity]).encode()).hexdigest()
    cache = Path(cache_dir).resolve()
    target = _inside(cache, cache/f'video_{key}.png')
    if target.is_file():
        return target
    times = sample_timestamps(info['duration'])
    with tempfile.TemporaryDirectory(prefix='h3lv-video-sheet-') as temporary:
        images = _sample_images(source, info, temporary, times, (320, 180))
    sheet = Image.new('RGB', (1280, 440), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), 'Motion reference samples | normalized 24 fps | labels are outside video frames',
              fill='white')
    for index, (image, timestamp) in enumerate(zip(images, times)):
        image.thumbnail((320, 180), Image.Resampling.LANCZOS)
        left, top = (index % 4) * 320, 24 + (index // 4) * 208
        sheet.paste(image, (left + (320 - image.width) // 2, top + (180 - image.height) // 2))
        draw.text((left + 8, top + 186),
                  f'Frame {round(timestamp * VIDEO_FPS)} | {timestamp:.3f} s', fill='white')
    if identity != _identity(source):
        raise ValueError('读取过程中参考视频已改变，请重试。')
    _atomic_bytes(target, _png(sheet))
    return target
