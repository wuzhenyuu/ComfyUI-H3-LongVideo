"""Local music analysis jobs and source-bound director-brief annotations."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

from . import core, controller

MODEL_NAME = 'OpenMOSS-Team/MOSS-Music-8B-Instruct'
MODEL_DIRECTORY = Path(__file__).resolve().parents[2]/'models'/'moss_music'/'MOSS-Music-8B-Instruct'
MODEL_REVISION = 'fce7f8304e96cc2d3398b8106456cbb2ecec3139'
RULE_VERSION = 2
TEXT_FIELDS = ('trend', 'summary', 'melody', 'rhythm', 'timbre', 'vocal', 'lyrics', 'uncertainty')
BLOCK_START = '<!-- h3lv:music-emotion:start -->'
BLOCK_END = '<!-- h3lv:music-emotion:end -->'
_BLOCK = re.compile(r'\n*'+re.escape(BLOCK_START)+r'.*?'+re.escape(BLOCK_END), re.S)
_RUNNING = None
_AUDIO_HASHES = {}


def _now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='milliseconds')


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _directory(root, project_id):
    return core.project_path(root, project_id)/'state'/'music_emotion'


def active(project_id=None):
    with core.LOCK:
        return _RUNNING is not None and (project_id is None or _RUNNING['job']['project_id'] == project_id)


def require_inactive(project_id=None):
    if active(project_id):
        raise ValueError('本机正在分析音乐情绪，请等待完成或取消后再继续。')


def _file_signature(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def model_signature():
    directory = MODEL_DIRECTORY.resolve()
    required = ('config.json', 'model.safetensors.index.json', 'preprocessor_config.json',
                'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json',
                'chat_template.jinja', 'configuration_moss_music.py', 'modeling_moss_music.py', 'processing_moss_music.py')
    missing = [name for name in required if not (directory/name).is_file() or not (directory/name).stat().st_size]
    if missing:
        raise ValueError(f'本地 MOSS-Music 模型尚未完整下载，缺少：{", ".join(missing)}。目录：{directory}')
    try:
        index = json.loads((directory/'model.safetensors.index.json').read_text(encoding='utf-8'))
        weights = index.get('weight_map')
        if not isinstance(weights, dict) or not weights:
            raise ValueError('模型权重索引为空。')
        shards = set(weights.values())
        if any(not isinstance(name, str) or Path(name).name != name or not name.endswith('.safetensors') for name in shards):
            raise ValueError('模型权重索引包含无效分片名。')
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError('模型权重索引损坏，请重新下载该文件。') from error
    missing = [name for name in sorted(shards) if not (directory/name).is_file() or not (directory/name).stat().st_size]
    if missing:
        raise ValueError('本地 MOSS-Music 模型尚未完整下载，缺少权重分片：'+', '.join(missing))
    files = sorted(path for path in directory.rglob('*') if path.is_file() and '.cache' not in path.parts
                   and not path.name.endswith(('.part', '.incomplete', '.tmp')))
    return hashlib.sha256(json.dumps([[path.relative_to(directory).as_posix(), *_file_signature(path)]
                                     for path in files], ensure_ascii=False).encode()).hexdigest()


def readiness():
    try:
        model_signature()
        missing = [name for name in ('torch', 'transformers', 'accelerate', 'soundfile', 'scipy', 'einops', 'safetensors', 'lmformatenforcer')
                   if importlib.util.find_spec(name) is None]
        if missing:
            raise ValueError('当前 ComfyUI Python 缺少依赖：'+', '.join(missing))
        if not Path(__file__).with_name('music_emotion_infer.py').is_file():
            raise ValueError('缺少音乐情绪推理 worker。')
        return True, ''
    except (ValueError, OSError, ImportError) as error:
        return False, str(error)


def _source(directory, plan):
    import soundfile as sf
    path = core.audio_file(directory, 'source.wav').resolve()
    signature = _file_signature(path)
    cached = _AUDIO_HASHES.get(str(path))
    if cached is None or cached['signature'] != signature:
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(1024*1024), b''):
                digest.update(block)
        if signature != _file_signature(path):
            raise ValueError('原音文件正在变化，请等待写入完成后重试。')
        cached = {'signature': signature, 'sha256': digest.hexdigest()}
        _AUDIO_HASHES[str(path)] = cached
    info = sf.info(str(path))
    if int(plan['sample_rate']) != info.samplerate or int(plan['samples']) != info.frames:
        raise ValueError('项目采样率或样本数与 source.wav 不一致，请重新分析音频。')
    return dict(path=str(path), sha256=cached['sha256'], signature=signature,
                sample_rate=info.samplerate, samples=info.frames)


def segment_input(source, row, signature):
    start, end = row['start_sample'], row['end_sample']
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= source['samples']:
        raise ValueError('音乐情绪分析的原音样本范围无效。')
    value = dict(source_sha256=source['sha256'], sample_rate=source['sample_rate'],
                 start_sample=start, end_sample=end, lyrics=str(row.get('text') or ''),
                 model_name=MODEL_NAME, model_revision=MODEL_REVISION,
                 model_signature=signature, rule_version=RULE_VERSION)
    key = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return value, key


def validate_analysis(value):
    if not isinstance(value, dict):
        raise ValueError('模型未返回结构化音乐情绪分析。')
    labels, intensity = value.get('labels'), value.get('intensity')
    if not isinstance(labels, list) or not labels or len(labels) > 12 or any(not isinstance(label, str) or not label.strip() or len(label) > 80 for label in labels):
        raise ValueError('模型情绪标签无效。')
    if any(BLOCK_START in label or BLOCK_END in label for label in labels):
        raise ValueError('模型输出含保留标记。')
    if isinstance(intensity, bool) or not isinstance(intensity, (int, float)) or not math.isfinite(intensity) or not 0 <= intensity <= 5:
        raise ValueError('模型情绪强度必须是 0 到 5 的主观估计。')
    result = {'labels': [label.strip() for label in labels], 'intensity': intensity}
    for name in TEXT_FIELDS:
        text = value.get(name)
        if not isinstance(text, str) or not text.strip() or len(text) > 6000:
            raise ValueError(f'模型音乐情绪字段 {name} 缺失或无效。')
        if BLOCK_START in text or BLOCK_END in text:
            raise ValueError('模型输出含保留标记。')
        result[name] = text.strip()
    return result


def strip_brief(text):
    return _BLOCK.sub('', str(text or '')).rstrip()


def annotate_brief(text, analysis):
    analysis = validate_analysis(analysis)
    lines = [BLOCK_START, '【音乐情绪分析】', '以下内容为本地模型的主观音乐表达估计，强度不是概率。',
             '情绪标签：'+'、'.join(analysis['labels']), f"情绪强度：{analysis['intensity']:g}/5（模型估计）",
             '情绪趋势：'+analysis['trend'], '表达概括：'+analysis['summary']]
    lines.extend(f'{label}：{analysis[name]}' for name, label in
                 (('melody', '旋律依据'), ('rhythm', '节奏依据'), ('timbre', '音色依据'),
                  ('vocal', '演唱依据'), ('lyrics', '歌词含义'), ('uncertainty', '不确定性')))
    return strip_brief(text)+'\n\n'+'\n'.join([*lines, BLOCK_END])


def invalidate(row, reason='原音区间或歌词已修改'):
    emotion = row.get('music_emotion')
    if not emotion:
        return False
    emotion.update(status='stale', stale_reason=reason)
    previous = row.get('prompt', '')
    row['prompt'] = strip_brief(previous)
    return previous != row['prompt']


def invalidate_changed(plan):
    changed = []
    for index, row in enumerate(plan['segments']):
        value = row.get('music_emotion') or {}
        source = value.get('input') or {}
        if value.get('status') == 'ready' and any(source.get(key) != current for key, current in
                (('start_sample', row['start_sample']), ('end_sample', row['end_sample']),
                 ('sample_rate', plan['sample_rate']), ('lyrics', str(row.get('text') or '')))):
            if invalidate(row):
                core.request_regeneration(row, '音乐情绪依据已修改')
                changed.append(index)
    return changed


def _progress(runtime, job):
    path = runtime['directory']/'response.json.progress.json'
    if not path.is_file():
        return
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        done = value.get('done')
        if type(done) is int and 0 <= done <= runtime['worker_total']:
            job['done'] = max(job['done'], runtime['cached_count']+done)
        index = value.get('index')
        if index is None or type(index) is int and index in runtime['indices']:
            job['index'] = index
    except (OSError, json.JSONDecodeError):
        pass


def status(root, project_id):
    directory = _directory(root, project_id)
    with core.LOCK:
        if _RUNNING is not None and _RUNNING['job']['project_id'] == project_id:
            runtime = _RUNNING
            job = copy.deepcopy(runtime['job'])
            _progress(runtime, job)
        elif (directory/'latest.json').is_file():
            job = json.loads((directory/'latest.json').read_text(encoding='utf-8'))
            if job.get('status') == 'running':
                job.update(status='failed', error='服务已重启；未完成的情绪分析未自动重发。')
        else:
            job = None
    ready, error = readiness()
    result = dict(ready=ready, model_name=MODEL_NAME, model_directory=str(MODEL_DIRECTORY), job=job)
    if error:
        result['error'] = error
    return result


def _persist(runtime):
    _write(runtime['directory']/'job.json', runtime['job'])
    _write(runtime['base']/'latest.json', runtime['job'])


def _queue_idle(server):
    running, pending = server.prompt_queue.get_current_queue()
    if running or pending:
        raise ValueError('ComfyUI 还有正在执行或排队的任务，请等待队列空闲后分析音乐情绪。')


def _prepare_gpu(server, cancel):
    queue = server.prompt_queue
    with queue.mutex:
        _queue_idle(server)
        queue.set_flag('unload_models', True)
    deadline = time.monotonic()+30
    while queue.get_flags(reset=False).get('unload_models'):
        if cancel.wait(.1):
            return False
        _queue_idle(server)
        if time.monotonic() > deadline:
            raise ValueError('ComfyUI 尚未释放缓存模型，请等待队列空闲后重试。')
    if cancel.wait(1):
        return False
    _queue_idle(server)
    return True


def start(root, project_id, revision, indices, server, force=False):
    global _RUNNING
    from .routes import require_idle
    root = Path(root)
    with core.LOCK:
        require_inactive()
        if controller.TASKS:
            raise ValueError('仍有 H3 顺序生成任务，请等待其全部结束后分析音乐情绪。')
        plan = core.read_plan(root, project_id)
        require_idle(plan)
        if type(revision) is not int or revision != plan.get('revision'):
            raise ValueError('项目已变化，请刷新后重新分析音乐情绪。')
        if not isinstance(indices, list) or not indices or any(type(i) is not int or not 0 <= i < len(plan['segments']) for i in indices) or len(set(indices)) != len(indices):
            raise ValueError('请选择有效且不重复的分段。')
        if type(force) is not bool:
            raise ValueError('force 必须为布尔值。')
        ready, error = readiness()
        if not ready:
            raise ValueError(error)
        _queue_idle(server)
        project = core.project_path(root, project_id)
        source, signature = _source(project, plan), model_signature()
        identifier = uuid.uuid4().hex
        base = _directory(root, project_id)
        job = dict(id=identifier, project_id=project_id, status='running', done=0, total=len(indices),
                   index=None, revision=revision, created_at=_now(), model_name=MODEL_NAME,
                   model_revision=MODEL_REVISION, model_directory=str(MODEL_DIRECTORY), source=source, items=[])
        runtime = dict(job=job, root=root, base=base, directory=base/'jobs'/identifier,
                       source=source, signature=signature, indices=list(indices), snapshot=copy.deepcopy(plan),
                       cancel=threading.Event(), process=None, cached_count=0, worker_total=0, attempted=0, server=server)
        for index in indices:
            row = plan['segments'][index]
            value, key = segment_input(source, row, signature)
            item = dict(index=index, key=key, input=value, status='pending')
            cached = base/'cache'/(key+'.json')
            if not force and cached.is_file():
                try:
                    saved = json.loads(cached.read_text(encoding='utf-8'))
                    if saved.get('key') == key and saved.get('input') == value:
                        item.update(analysis=validate_analysis(saved['analysis']), raw_text=saved.get('raw_text', ''), status='cached')
                        runtime['cached_count'] += 1
                except (ValueError, KeyError, OSError):
                    pass
            job['items'].append(item)
        runtime['worker_total'] = len(indices)-runtime['cached_count']
        job['done'] = runtime['cached_count']
        _persist(runtime)
        _RUNNING = runtime
        threading.Thread(target=_run, args=(runtime,), name='h3lv-music-emotion', daemon=True).start()
    return status(root, project_id)


def _commit(runtime):
    with core.LOCK:
        if _RUNNING is not runtime or runtime['cancel'].is_set():
            return
        job = runtime['job']
        plan = core.read_plan(runtime['root'], job['project_id'])
        snapshot = runtime['snapshot']
        current_source = _source(core.project_path(runtime['root'], plan['id']), plan)
        valid = (plan.get('revision') == job['revision'] and plan.get('run_status') == snapshot.get('run_status')
                 and plan['id'] == snapshot['id'] and current_source['sha256'] == runtime['source']['sha256']
                 and model_signature() == runtime['signature'])
        changed = []
        for item in job['items']:
            if item['status'] not in ('cached', 'succeeded'):
                continue
            index = item['index']
            if not valid or not 0 <= index < len(plan['segments']):
                item['apply_status'] = 'stale'
                continue
            row, old_row = plan['segments'][index], snapshot['segments'][index]
            value, key = segment_input(current_source, row, runtime['signature'])
            if key != item['key'] or value != item['input'] or any(row.get(name) != old_row.get(name) for name in ('prompt', 'final_prompt')):
                item['apply_status'] = 'stale'
                continue
            analysis = validate_analysis(item['analysis'])
            prompt = annotate_brief(row.get('prompt', ''), analysis)
            emotion = dict(status='ready', analysis=analysis, key=key, input=value,
                           source=runtime['source'], model_directory=str(MODEL_DIRECTORY),
                           model_name=MODEL_NAME, model_revision=MODEL_REVISION,
                           raw_text=item.get('raw_text', ''), job_id=job['id'], analyzed_at=_now())
            previous = row.get('music_emotion') or {}
            content_changed = (row.get('prompt', '') != prompt or previous.get('key') != key
                               or previous.get('analysis') != analysis or previous.get('status') != 'ready')
            if content_changed:
                row.update(prompt=prompt, music_emotion=emotion)
                core.request_regeneration(row, '音乐情绪分析已更新导演简报')
                changed.append(index)
            item['apply_status'] = 'applied'
        if changed:
            plan.update(revision=plan.get('revision', 0)+1, approved=False, run_status='draft', changed_segments=changed)
            if plan.get('final_video'):
                plan['final_stale'] = True
            core.write_plan(runtime['root'], plan)
        job['changed_segments'] = changed


def _read_response(runtime, path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema_version') != 1 or value.get('model_revision') != MODEL_REVISION or not isinstance(value.get('segments'), list):
        raise ValueError('情绪 worker 返回了不匹配的格式或模型版本。')
    pending = {item['index']: item for item in runtime['job']['items'] if item['status'] == 'pending'}
    seen = set()
    for result in value['segments']:
        index = result.get('index')
        if type(index) is not int or index not in pending or index in seen:
            raise ValueError('情绪 worker 返回了未请求或重复的分段。')
        seen.add(index)
    for result in value['segments']:
        item = pending[result['index']]
        try:
            item.update(analysis=validate_analysis(result.get('analysis')), raw_text=str(result.get('raw_text') or ''), status='succeeded')
            _write(runtime['base']/'cache'/(item['key']+'.json'), item)
        except ValueError as error:
            item.update(status='failed', error=str(error))
    for index, item in pending.items():
        if index not in seen:
            item.update(status='failed', error='模型没有返回本段的有效情绪分析。')
    if value.get('errors'):
        runtime['job']['worker_errors'] = value['errors']
    errors = {error['index'] for error in value.get('errors', []) if isinstance(error, dict)
              and type(error.get('index')) is int and error['index'] in pending}
    runtime['attempted'] = len(seen | errors)


def _run(runtime):
    global _RUNNING
    job = runtime['job']
    try:
        pending = [item for item in job['items'] if item['status'] == 'pending']
        if pending:
            if not _prepare_gpu(runtime['server'], runtime['cancel']):
                return
            request = dict(model_path=str(MODEL_DIRECTORY.resolve()), model_revision=MODEL_REVISION,
                           audio_path=runtime['source']['path'], max_new_tokens=1536,
                           segments=[dict(index=item['index'], start_sample=item['input']['start_sample'],
                                          end_sample=item['input']['end_sample'], sample_rate=item['input']['sample_rate'],
                                          lyrics=item['input']['lyrics']) for item in pending])
            request_path, response_path = runtime['directory']/'request.json', runtime['directory']/'response.json'
            _write(request_path, request)
            with (runtime['directory']/'worker.log').open('w', encoding='utf-8') as log:
                with core.LOCK:
                    if runtime['cancel'].is_set():
                        return
                    runtime['process'] = subprocess.Popen([sys.executable, str(Path(__file__).with_name('music_emotion_infer.py')),
                        '--request', str(request_path), '--response', str(response_path)], stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                code = runtime['process'].wait()
            if runtime['cancel'].is_set():
                return
            if response_path.is_file():
                _read_response(runtime, response_path)
            if code:
                job['error'] = f'本地情绪 worker 退出（代码 {code}），已保留有效的部分结果；详情见本项目分析日志。'
            elif not response_path.is_file():
                raise ValueError('本地情绪 worker 未输出分析结果；请检查依赖、模型及显存。')
        _commit(runtime)
    except Exception as error:
        job['error'] = str(error)
    finally:
        with core.LOCK:
            cancelled = runtime['cancel'].is_set()
            for item in job['items']:
                if item['status'] == 'pending':
                    item.update(status='cancelled' if cancelled else 'failed', error=job.get('error', '未完成分析'))
            job['done'] = runtime['cached_count']+runtime['attempted']
            _progress(runtime, job)
            failed = any(item['status'] == 'failed' or item.get('apply_status') == 'stale' for item in job['items'])
            job.update(status='cancelled' if cancelled else 'failed' if failed or job.get('error') else 'completed', finished_at=_now())
            if any(item.get('apply_status') == 'stale' for item in job['items']):
                job['error'] = '项目依据已变化，旧范围分析仅保留为候选，未覆盖当前导演简报。'
            try:
                _persist(runtime)
            finally:
                if _RUNNING is runtime:
                    _RUNNING = None


def cancel(root, project_id):
    with core.LOCK:
        if _RUNNING is not None and _RUNNING['job']['project_id'] == project_id:
            runtime = _RUNNING
            runtime['cancel'].set()
            process = runtime['process']
            if process is not None and process.poll() is None:
                process.terminate()
    return status(root, project_id)
