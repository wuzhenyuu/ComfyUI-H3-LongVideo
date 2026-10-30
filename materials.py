"""Resolved, ordered segment materials shared by images and prompt expansion."""
import hashlib
import json
from pathlib import Path


def videos_enabled(plan):
    return (plan.get('video_materials_version') == 1
            or any(key in plan for key in ('default_video_refs', 'default_video_note'))
            or any(any(key in row for key in ('video_refs', 'video_note', 'video_reference_source'))
                   for row in plan.get('segments', [])))


def effective(plan, row, directory=None, require=False):
    from .core import normalize_reference_names
    source = row.get('reference_source', 'custom' if row.get('refs') else 'default')
    if source not in ('default', 'custom'):
        raise ValueError('参考图来源无效。')
    names = plan.get('default_refs', []) if source == 'default' else row.get('refs', [])
    note = plan.get('default_material_note', '') if source == 'default' else row.get('material_note', '')
    if 'shot_request' in row:
        note = row['shot_request']
    names = normalize_reference_names(names, directory)
    kind = row.get('visual_type', 'performance')
    if kind not in ('performance', 'atmosphere', 'environment'):
        raise ValueError('画面类型无效。')
    if require and not names:
        raise ValueError('本段没有有效参考图，请上传默认图或本段自定义图。')
    return {'refs': names, 'material_note': str(note).strip(), 'visual_type': kind}


def packet(plan, row, directory):
    from .core import reference_directory, inside
    value = effective(plan, row, directory, require=True)
    paths = [inside(reference_directory(directory), reference_directory(directory)/name) for name in value['refs']]
    videos = {}
    if videos_enabled(plan):
        from .video_materials import packet as video_packet
        videos = video_packet(plan, row, directory)
    # Explicit visual requests supersede historical automatic direction. Keep
    # deliberate director edits separate so they are not lost with that fallback.
    from .core import segment_brief
    intent = '' if 'shot_request' in row else str(row.get('shot_intent') or '').strip()
    explicit = bool(value['material_note'] or intent)
    brief = '' if explicit else segment_brief(plan, row, row.get('camera_start', 'medium shot'),
                                             row.get('camera_end', 'medium shot'), row.get('camera_move', ''), '')
    director = {}
    user_brief = str(row.get('prompt') or '')
    if row.get('brief_source') == 'user' and user_brief.strip():
        director['user_director_brief'] = user_brief
    emotion = {}
    if row.get('music_emotion', {}).get('status') == 'ready':
        from .music_emotion import validate_analysis
        emotion['music_emotion'] = validate_analysis(row['music_emotion']['analysis'])
    return {**value, **videos, **emotion, **director, 'project_id': plan['id'], 'segment_index': row['index'],
            'contract_version': 3 if 'shot_request' in row else 2,
            'shot_intent': intent,
            'brief_source': 'none' if explicit else 'automatic',
            'mode': plan['mode'], 'brief': brief, 'duration': row['duration'],
            'generation_frames': row['generation_frames'],
            'generation_seconds': row['generation_frames']/24,
            'audio_role': row.get('audio_role', 'uncertain'),
            'audio_section': row.get('audio_section', ''),
            'audio_role_reason': row.get('audio_role_reason', ''),
            'paths': [str(p) for p in paths],
            'hashes': [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
            'cache_dir': str(Path(directory)/'cache'/'expansion'),
            'expanded_prompt': row.get('expanded_prompt'), 'expanded_key': row.get('expanded_key')}


def images(material):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    result = []
    for path in material['paths']:
        with Image.open(path) as im:
            value = ImageOps.exif_transpose(im).convert('RGB')
            result.append(torch.from_numpy(np.asarray(value).astype(np.float32)/255).unsqueeze(0))
    return tuple(result + [None]*(6-len(result)))


def stamp(plan, row):
    value = effective(plan, row)
    if videos_enabled(plan):
        from .video_materials import effective as effective_videos
        videos = effective_videos(plan, row)
        if videos['refs'] or videos['note'] or videos['source'] == 'custom':
            value['videos'] = videos
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
