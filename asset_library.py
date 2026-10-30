"""Shared image/video assets with immutable versions and project snapshots."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import uuid

from PIL import Image, ImageOps, UnidentifiedImageError

from .core import (MAX_REFERENCE_BYTES, MAX_SEGMENT_REFERENCES,
                   REFERENCE_IMAGE_EXTENSIONS, inside)
from . import video_materials

LOCK = threading.RLock()
FORMATS = {'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp', 'BMP': '.bmp', 'GIF': '.gif'}
METADATA_FIELDS = {'category', 'name', 'description', 'tags'}


def _now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='microseconds')


def _path(root, *parts):
    return inside(root, Path(root).joinpath(*parts))


def _asset_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{32}', value):
        raise ValueError('素材编号无效。')
    return value


def _version_number(value):
    if isinstance(value, bool) or not re.fullmatch(r'[1-9][0-9]*', str(value)):
        raise ValueError('素材图片版本无效。')
    return int(value)


def _read_index(root):
    path = _path(root, 'index.json')
    if not path.exists():
        return {'schema': 1, 'revision': 0, 'assets': {}}
    value = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(value, dict) or value.get('schema') != 1
            or type(value.get('revision')) is not int or value['revision'] < 0
            or not isinstance(value.get('assets'), dict)):
        raise ValueError('素材库清单无效，请检查 index.json。')
    return value


def _atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('xb') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, value):
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def _check_revision(index, expected_revision):
    if expected_revision is None:
        return
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError('素材库 revision 无效。')
    if expected_revision != index['revision']:
        raise ValueError('素材库已被其他窗口修改，请刷新后重试。')


def _asset(index, asset_id):
    value = index['assets'].get(_asset_id(asset_id))
    if value is None:
        raise FileNotFoundError('素材不存在，请刷新素材库。')
    return value


def _version(asset, number=None):
    number = asset['version'] if number is None else _version_number(number)
    for value in asset['versions']:
        if value['version'] == number:
            return value
    raise FileNotFoundError('素材图片版本不存在。')


def _image_path(root, version, thumbnail=False):
    digest = version['sha256']
    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise ValueError('素材图片哈希无效。')
    if thumbnail:
        return _path(root, 'thumbnails', f'{digest}.png')
    if version.get('media_type', 'image') == 'video':
        extension = version.get('extension')
        if extension not in video_materials.VIDEO_EXTENSIONS:
            raise ValueError('素材视频格式无效。')
        return _path(root, 'blobs', digest + extension)
    if version.get('format') not in FORMATS:
        raise ValueError('素材图片格式无效。')
    return _path(root, 'blobs', digest + FORMATS[version['format']])


def _public_asset(index, asset):
    result = copy.deepcopy(asset)
    result.setdefault('media_type', 'image')
    result.update({key: value for key, value in _version(result).items() if key != 'created_at'})
    digest = result['sha256']
    result['duplicate_ids'] = [item['id'] for item in index['assets'].values()
                               if item['id'] != asset['id']
                               and any(row['sha256'] == digest for row in item['versions'])]
    return result


def _metadata(category, name, description, tags, media_type='image'):
    allowed = ('video',) if media_type == 'video' else ('character', 'scene')
    if category not in allowed:
        raise ValueError('视频素材分类必须为 video。' if media_type == 'video'
                         else '图片素材分类只支持 character 或 scene。')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('素材名称不能为空。')
    if not isinstance(description, str):
        raise ValueError('素材说明必须是文字。')
    if tags is None:
        tags = []
    if not isinstance(tags, (list, tuple)) or any(not isinstance(tag, str) for tag in tags):
        raise ValueError('素材标签必须是文字列表。')
    return {'category': category, 'name': name.strip(), 'description': description.strip(),
            'tags': list(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))}


def _decoded_image(content):
    with Image.open(io.BytesIO(content)) as image:
        image_format = image.format
        if image_format not in FORMATS:
            raise ValueError('图片实际格式不受支持。')
        image.verify()
    with Image.open(io.BytesIO(content)) as image:
        normalized = ImageOps.exif_transpose(image).convert('RGBA' if 'A' in image.getbands()
                                                         or 'transparency' in image.info else 'RGB')
        normalized.load()
    return image_format, normalized


def _png_bytes(image):
    output = io.BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


def _prepare_upload(filename, content):
    if (not isinstance(filename, str) or not filename or Path(filename).name != filename
            or any(char in filename for char in '<>:"/\\|?*\0')):
        raise ValueError('素材图片文件名无效。')
    suffix = Path(filename).suffix.lower()
    if suffix in video_materials.VIDEO_EXTENSIONS:
        return _prepare_video(filename, content)
    if suffix not in REFERENCE_IMAGE_EXTENSIONS:
        raise ValueError('参考图只支持 PNG、JPG、WEBP、BMP 或 GIF 格式。')
    if not isinstance(content, (bytes, bytearray, memoryview)) or not content:
        raise ValueError('上传的参考图内容为空。')
    if len(content) > MAX_REFERENCE_BYTES:
        raise ValueError('单张参考图不能超过 32 MiB。')
    content = bytes(content)
    try:
        image_format, normalized = _decoded_image(content)
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise ValueError('上传内容不是有效图片。') from None
    value = {'sha256': hashlib.sha256(content).hexdigest(), 'original_filename': filename,
             'media_type': 'image',
             'format': image_format, 'width': normalized.width, 'height': normalized.height,
             'bytes': len(content)}
    normalized.thumbnail((256, 256), Image.Resampling.LANCZOS)
    return content, value, _png_bytes(normalized)


def _prepare_video(filename, content):
    if not isinstance(content, (bytes, bytearray, memoryview)) or not content:
        raise ValueError('上传的参考视频内容为空。')
    if len(content) > video_materials.MAX_VIDEO_BYTES:
        raise ValueError('单个参考视频不能超过 512 MiB。')
    content = bytes(content)
    with tempfile.TemporaryDirectory(prefix='h3lv-library-video-') as temporary:
        source = Path(temporary)/filename
        source.write_bytes(content)
        metadata = video_materials.validate_group([source])[0]
        thumbnail = video_materials.thumbnail(source)
    value = {'sha256': hashlib.sha256(content).hexdigest(), 'original_filename': filename,
             'media_type': 'video', 'extension': Path(filename).suffix.lower(),
             'bytes': len(content), **metadata}
    return content, value, thumbnail


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_bytes(path, content):
    if path.exists():
        if path.stat().st_size != len(content) or _file_hash(path) != hashlib.sha256(content).hexdigest():
            raise ValueError('素材图片文件与已有版本冲突，请检查素材库文件。')
    else:
        _atomic_bytes(path, content)


def list_assets(root, category=None, query='', include_deleted=False, media_type=None):
    if category is not None and category not in ('character', 'scene', 'video'):
        raise ValueError('素材分类只支持 character、scene 或 video。')
    if media_type is not None and media_type not in ('image', 'video'):
        raise ValueError('素材类型只支持 image 或 video。')
    query = str(query or '').strip().casefold()
    with LOCK:
        index = _read_index(root)
        assets = []
        for asset in index['assets'].values():
            if (not include_deleted and asset['deleted_at']) or (category and asset['category'] != category):
                continue
            if media_type and asset.get('media_type', 'image') != media_type:
                continue
            text = ' '.join([asset['name'], asset['description'], *asset['tags']]).casefold()
            if query and query not in text:
                continue
            assets.append(_public_asset(index, asset))
        assets.sort(key=lambda item: (item['updated_at'], item['id']), reverse=True)
        return {'revision': index['revision'], 'assets': assets}


def upload(root, filename, content, category='character', name='', description='', tags=None,
           expected_revision=None, asset_id=None):
    content, version, thumbnail = _prepare_upload(filename, content)
    media_type = version['media_type']
    if media_type == 'video':
        category = 'video'
    metadata = None if asset_id is not None else _metadata(
        category, name or Path(filename).stem, description, tags, media_type)
    with LOCK:
        index = _read_index(root)
        _check_revision(index, expected_revision)
        now = _now()
        if asset_id is None:
            asset = {'id': uuid.uuid4().hex, **metadata, 'media_type': media_type,
                     'version': 0, 'versions': [],
                     'created_at': now, 'updated_at': now, 'deleted_at': None}
            index['assets'][asset['id']] = asset
        else:
            asset = _asset(index, asset_id)
            if asset['deleted_at']:
                raise ValueError('素材已在回收站，请先恢复再上传新版本。')
            if asset.get('media_type', 'image') != media_type:
                raise ValueError('不能跨素材类型替换图片或视频，请新建素材。')
        version.update(version=asset['version'] + 1, created_at=now)
        _immutable_bytes(_image_path(root, version), content)
        thumb_path = _image_path(root, version, thumbnail=True)
        if not thumb_path.exists():
            _atomic_bytes(thumb_path, thumbnail)
        asset['versions'].append(version)
        asset.update(version=version['version'], updated_at=now)
        index['revision'] += 1
        _write_json(_path(root, 'index.json'), index)
        return {'revision': index['revision'], 'asset': _public_asset(index, asset)}


def edit_asset(root, asset_id, expected_revision=None, **fields):
    if fields.keys() - METADATA_FIELDS:
        raise ValueError('只能编辑素材名称、说明、分类和标签。')
    with LOCK:
        index = _read_index(root)
        _check_revision(index, expected_revision)
        asset = _asset(index, asset_id)
        metadata = _metadata(**{key: fields.get(key, asset[key]) for key in METADATA_FIELDS},
                             media_type=asset.get('media_type', 'image'))
        if any(asset[key] != value for key, value in metadata.items()):
            asset.update(metadata, updated_at=_now())
            index['revision'] += 1
            _write_json(_path(root, 'index.json'), index)
        return {'revision': index['revision'], 'asset': _public_asset(index, asset)}


def set_deleted(root, asset_id, deleted=True, expected_revision=None):
    if not isinstance(deleted, bool):
        raise ValueError('回收站状态必须是布尔值。')
    with LOCK:
        index = _read_index(root)
        _check_revision(index, expected_revision)
        asset = _asset(index, asset_id)
        if bool(asset['deleted_at']) != deleted:
            now = _now()
            asset.update(deleted_at=now if deleted else None, updated_at=now)
            index['revision'] += 1
            _write_json(_path(root, 'index.json'), index)
        return {'revision': index['revision'], 'asset': _public_asset(index, asset)}


def asset_path(root, asset_id, version=None, thumbnail=False):
    with LOCK:
        record = _version(_asset(_read_index(root), asset_id), version)
        original = _image_path(root, record)
        if not original.is_file():
            raise FileNotFoundError('素材原图已丢失。')
        if not thumbnail:
            return original
        target = _image_path(root, record, thumbnail=True)
        if not target.is_file():
            if record.get('media_type', 'image') == 'video':
                content = video_materials.thumbnail(original)
            else:
                _, image = _decoded_image(original.read_bytes())
                image.thumbnail((256, 256), Image.Resampling.LANCZOS)
                content = _png_bytes(image)
            _atomic_bytes(target, content)
        return target


def import_assets(root, project_dir, selection, media_type='image'):
    """Copy ordered PNG snapshots; the caller owns the project lock and plan update."""
    if not isinstance(selection, (list, tuple)):
        raise ValueError('请选择按顺序排列的素材列表。')
    if media_type not in ('image', 'video'):
        raise ValueError('素材类型只支持 image 或 video。')
    maximum = 3 if media_type == 'video' else MAX_SEGMENT_REFERENCES
    if len(selection) > maximum:
        raise ValueError('每段最多 3 个参考视频。' if media_type == 'video'
                         else f'每段最多 {MAX_SEGMENT_REFERENCES} 张参考图。')
    project = Path(project_dir).resolve()
    root = Path(root).resolve()
    if project.is_relative_to(root) or root.is_relative_to(project):
        raise ValueError('素材库与项目目录必须相互独立。')
    if not project.is_dir():
        raise FileNotFoundError('项目目录不存在。')
    snapshots = []
    seen = set()
    with LOCK:
        index = _read_index(root)
        for item in selection:
            if not isinstance(item, dict):
                raise ValueError('素材选择项无效。')
            asset = _asset(index, item.get('asset_id'))
            if asset['deleted_at']:
                raise ValueError('素材已在回收站，请先恢复再导入项目。')
            if asset.get('media_type', 'image') != media_type:
                raise ValueError('所选素材类型与目标参考类型不一致。')
            record = _version(asset, item.get('version'))
            key = (asset['id'], record['version'])
            if key in seen:
                raise ValueError('不能重复选择同一素材的同一图片版本。')
            seen.add(key)
            snapshots.append((copy.deepcopy(asset), copy.deepcopy(record), _image_path(root, record)))
    if not snapshots:
        if media_type == 'video':
            return {'video_refs': [], 'video_note': '', 'video_metadata': {}, 'origins': {}}
        return {'refs': [], 'origins': {}}
    sources_path = inside(project, project/'state'/'library_sources.json')
    if sources_path.exists():
        sources = json.loads(sources_path.read_text(encoding='utf-8'))
        if not isinstance(sources, dict) or sources.get('schema') != 1 or not isinstance(sources.get('origins'), dict):
            raise ValueError('项目素材来源记录无效。')
    else:
        sources = {'schema': 1, 'origins': {}}
    if media_type == 'video':
        return _import_videos(project, snapshots, sources_path, sources)
    prepared = []
    origins = {}
    for asset, record, original in snapshots:
        content = original.read_bytes()
        if hashlib.sha256(content).hexdigest() != record['sha256']:
            raise ValueError('素材原图已改变，无法导入该历史版本。')
        _, image = _decoded_image(content)
        name = f"lib_{asset['id']}_v{record['version']}_{uuid.uuid4().hex}.png"
        prepared.append((inside(project, project/'refs'/name), _png_bytes(image)))
        origins[name] = {'asset_id': asset['id'], 'version': record['version'],
                         'name_at_import': asset['name'], 'category': asset['category'],
                         'description': asset['description'], 'sha256': record['sha256'],
                         'imported_at': _now()}
    written = []
    try:
        for path, content in prepared:
            _immutable_bytes(path, content)
            written.append(path)
        sources['origins'].update(origins)
        _write_json(sources_path, sources)
    except (OSError, ValueError):
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return {'refs': [path.name for path, _ in prepared], 'origins': origins}


def _import_videos(project, snapshots, sources_path, sources):
    # Validate every selection before writing any project file.
    values = video_materials.validate_group([original for _, _, original in snapshots])
    prepared, origins, metadata, notes = [], {}, {}, []
    for (asset, record, original), value in zip(snapshots, values):
        if _file_hash(original) != record['sha256']:
            raise ValueError('素材原视频已改变，无法导入该历史版本。')
        name = f"lib_{asset['id']}_v{record['version']}_{uuid.uuid4().hex}{record['extension']}"
        prepared.append((video_materials.path(project, name), original, record['sha256']))
        metadata[name] = value
        notes.append(f"{asset['name']}：{asset['description']}" if asset['description'] else asset['name'])
        origins[name] = {'asset_id': asset['id'], 'version': record['version'],
                         'name_at_import': asset['name'], 'category': 'video', 'media_type': 'video',
                         'description': asset['description'], 'sha256': record['sha256'],
                         'imported_at': _now(), **value}
    written = []
    try:
        for target, original, expected_hash in prepared:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.tmp')
            try:
                with original.open('rb') as source, temporary.open('xb') as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                if _file_hash(temporary) != expected_hash:
                    raise ValueError('素材原视频已改变，无法导入该历史版本。')
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            written.append(target)
        sources['origins'].update(origins)
        _write_json(sources_path, sources)
    except (OSError, ValueError):
        for target in written:
            target.unlink(missing_ok=True)
        raise
    return {'video_refs': [target.name for target, _, _ in prepared],
            'video_note': '\n'.join(notes), 'video_metadata': metadata, 'origins': origins}
