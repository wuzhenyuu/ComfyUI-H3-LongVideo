#!/usr/bin/env python3
"""Self-contained MiniMax H3 grammar plus the H3 LongVideo segment contract.

Vendored from minimax-h3-prompt-writing/scripts/validate_h3_prompt.py,
local skill revision 2026-10-28; source SHA256:
582528fabc0b62b2f3a20c7e8d7cbf5f625b4e6fa47609e4e591f10320ec0603.
The source skill cites MiniMax-H3 base-en.txt/ref-en.txt. Chinese prose and
LongVideo's one-shot, N/A sound, no-overlay and audio-slot gates are local
requirements. Runtime imports never depend on an installed skill or its path.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


BASE_FIELDS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)
REF_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
REF_LABEL_RE = re.compile(r"<(?:Subject|Picture|Video|Audio) \d+>")
REF_DEFINITION_RE = re.compile(
    r"^(<(?P<kind>Subject|Picture|Video|Audio) (?P<number>\d+)>)\s*(?:是|[:：])"
)
SHOT_RE = re.compile(r"\[Shot (\d+)\](?: At (\d{2}):(\d{2}\.\d{3}),)?")
TASK_TYPES = {
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
}
VISIBLE_MARKERS = {
    "fully_preserved",
    "partially_preserved",
    "attribute_transfer",
    "weak_reference",
}
AUDIO_MARKERS = {
    "fully_copy",
    "partially_copy",
    "reference",
    "weak_reference",
}
NO_OVERLAY = '所有画面均不新增字幕、标题、歌词、水印或图形叠加；不转录参考图背景中的文字。'


def validate_video_material(material):
    """Check the verified ordered video packet, without probing media again."""
    paths = material.get('video_paths', [])
    if not isinstance(paths, (list, tuple)) or len(paths) > 3 or any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError('参考视频 video_paths 必须是最多 3 段的有效路径列表。')
    count = len(paths)
    for field in ('video_hashes', 'video_durations'):
        values = material.get(field, [])
        if not isinstance(values, (list, tuple)) or len(values) != count:
            raise ValueError(field+' 必须与 video_paths 数量和顺序一致；缺少真实视频元数据。')
    if any(not isinstance(value, str) or not value.strip() for value in material.get('video_hashes', [])):
        raise ValueError('参考视频缺少有效的 video_hashes。')
    if 'video_refs' in material:
        refs = material['video_refs']
        if not isinstance(refs, (list, tuple)) or len(refs) != count:
            raise ValueError('video_refs 必须与 video_paths 数量和顺序一致。')
    durations = material.get('video_durations', [])
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           or not 2 <= value <= 15 for value in durations):
        raise ValueError('每段参考视频的实际时长必须为 2–15 秒。')
    if sum(durations) > 15+1e-6:
        raise ValueError('参考视频实际总时长不得超过 15 秒。')
    note = str(material.get('video_note', '') or '')
    mentions = re.findall(r'@视频\s*(\d+)|<Video\s+(\d+)>', note)
    if any(not 1 <= int(ui or label) <= count for ui, label in mentions):
        raise ValueError('视频用途说明引用了不存在的 @视频 或 Video 输入槽。')
    return count


def validate_material(material):
    """Reject unsupported generation lengths before a provider is called."""
    seconds = material.get('generation_seconds')
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 4 <= seconds <= 15:
        raise ValueError('实际 generation_seconds 必须在 4–15 秒内；请先调整分段，不能用原始音频时长替代生成时长。')
    frames = material.get('generation_frames')
    if frames is not None and (isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0 or abs(frames/24-seconds) > 1e-6):
        raise ValueError('generation_seconds 必须与实际 generation_frames / 24 一致。')
    if material.get('visual_type') not in ('performance', 'atmosphere', 'environment'):
        raise ValueError('扩写素材的 visual_type 无效。')
    validate_video_material(material)
    return seconds


def _positive_vocal_requirement(text):
    terms = r'口型|嘴部开合|嘴巴开合|口部动作|嘴唇张开|张嘴|开口|演唱|唱歌|跟唱|说话|朗读|念白|lip[- ]?sync|sing(?:ing)?|speak(?:ing)?'
    for clause in re.split(r'[，,。；;！!？?\n]', text):
        for match in re.finditer(terms, clause, re.IGNORECASE):
            before = clause[:match.start()]
            if not re.search(r'(?:不|未|禁止|无需|避免|没有|不得|切勿)[^，。；\n]{0,24}$|\b(?:no|not|never|without)\b[^,.;\n]{0,24}$', before, re.IGNORECASE):
                return True
    return False


def _positive_video_audio_requirement(text):
    """Video samples carry visual motion only, even if the source has audio."""
    source = r'(?:<Video\s+\d+>|(?:参考|源|原)视频|视频)'
    audio = r'<Audio\s+\d+>|原声|原音轨|源音轨|源音频|音色|歌词|声轨|音轨|伴奏|人声|对白|声音|背景音乐|音乐|配乐|BGM|音频|音效|歌声|嗓音'
    # Bind the audible feature to its actual source. A shot may mention Video
    # motion and the separate project's Audio 1 together without borrowing the
    # video's sound, so co-occurrence in a clause is not a provenance claim.
    link = r'(?:\s|中的|里的|中|里|的|所含|附带|自带|原有|原始|同步|现场|提供|提取|保持|保留|沿用|复制|使用|复用|作为|用于|是|为|参考自|参考|依据|来自|取自|源于|选自|对应|映射|[（()）：:]){0,20}'
    negative = r'(?:不|无|未|禁止|禁用|无需|避免|没有|不得|切勿)[^，。；\n]{0,32}$|\b(?:no|not|never|without)\b[^,.;\n]{0,32}$'
    for clause in re.split(r'[，,。；;！!？?\n]', text):
        for match in re.finditer(audio, clause, re.IGNORECASE):
            if not (re.search(source+link+r'$', clause[:match.start()])
                    or re.match(link+source, clause[match.end():])):
                continue
            if re.match(r'\s*(?:均|完全|一律)?(?:不参与|不参考|不复用|不保留|不复制|不使用|禁用|关闭|不启用|不传入)', clause[match.end():]):
                continue
            if not re.search(negative, clause[:match.start()], re.IGNORECASE):
                return True
    return False


def _overlong_video_claim(text, material):
    """Do not promise source footage that the native frame input cannot hold."""
    for number, seconds in enumerate(material.get('video_durations', []), 1):
        if seconds <= material['generation_seconds']:
            continue
        for clause in re.split(r'[。；;！!？?\n]', text):
            if f'<Video {number}>' not in clause:
                continue
            if not re.search(r'完整使用|完整复刻|完整呈现|完整覆盖|从头到尾|全部源帧|整段源视频|整个源视频', clause):
                continue
            if not re.search(r'(?:不|未|禁止|无需|避免|没有|不得|切勿|不能|不承诺)[^。；\n]{0,24}(?:完整|从头到尾|全部|整段|整个)', clause):
                return True
    return False


def validate_segment_prompt(text, count, material=None):
    """Strict Ref2VA syntax; metadata adds duration and visual/audio checks."""
    duration = validate_material(material) if material is not None else None
    report = Report()
    _check_placeholders(text, report)
    _check_language(text, report)
    _check_prompt_start(text, 'ref2va', report)
    _check_fields(text, REF_FIELDS, report)
    _check_ref2va(text, report)
    sections = {name: _extract_section(text, name, REF_FIELDS) for name in REF_FIELDS}
    details = sections['detailed_description']
    _check_dialogue(text, details, report)
    _check_shots(details, duration if duration is not None else float('inf'), report)
    if len(list(SHOT_RE.finditer(details))) != 1:
        report.errors.append('每段扩写必须只有一个连续的 [Shot 1]。')
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 6:
        report.errors.append('本插件需要 1–6 张有效参考图。')
    labels = re.findall(r'<(?:Subject|Picture|Video|Audio)\b[^<>]*>', text)
    if any(not re.fullmatch(r'<(?:Subject|Picture|Video|Audio) [1-9]\d*>', label) for label in labels):
        report.errors.append('参考标签必须使用原格式和正整数编号。')
    pictures = re.findall(r'<Picture (\d+)>', text)
    if not pictures or any(not 1 <= int(number) <= count for number in pictures):
        report.errors.append('扩写结果引用了不存在的图片，或缺少实际 Picture 来源。')
    definitions = sections['subject_definitions']
    subjects = set(re.findall(r'(?m)^(<Subject [1-9]\d*>)\s*(?:是|[:：])', definitions))
    if not subjects or set(re.findall(r'<Subject \d+>', text))-subjects:
        report.errors.append('每个 Subject 必须有独立声明，且从 <Subject 1> 连续编号。')
    video_count = len(material.get('video_paths', [])) if material is not None else 0
    videos = set(re.findall(r'<Video (\d+)>', text))
    if videos and (not video_count or any(not 1 <= int(number) <= video_count for number in videos)):
        report.errors.append('不能虚构 Video 素材；Video 编号必须对应真实 video_paths 输入槽。')
    if re.search(r'\bvideo (?:editing|continuation)\b', sections['summary']):
        report.errors.append('参考视频仅提供视觉动作和运镜，不是 video editing 或 video continuation 任务。')
    if video_count:
        if not re.match(r'\[[^\]]*\breference generation\b[^\]]*\]', sections['summary']):
            report.errors.append('视频视觉参考的 summary 必须声明 reference generation。')
        defined_sources = set(re.findall(r'<Video (\d+)>', definitions))
        if defined_sources != {str(number) for number in range(1, video_count+1)}:
            report.errors.append('每段选定参考视频都必须在 subject_definitions 中声明职责或作为 Subject 动作来源。')
        for line in definitions.splitlines():
            if not re.search(r'<Video \d+>', line):
                continue
            if line.startswith('<Video '):
                if not re.search(r'运镜|镜头|节奏|时序|摄影机|摄像机', line):
                    report.errors.append('独立 Video 定义只能承担整段运镜、镜头节奏或时序结构。')
            elif line.startswith('<Subject ') and not re.search(r'动作|姿态|步态|肢体|表情|视线|目光', line):
                report.errors.append('Subject 从视频借用的内容必须明确是视觉动作，人物与场景仍由图片定义。')
        if re.search(r'\baudio reuse\b|fully_copy|partially_copy', text):
            report.errors.append('参考视频只用于视觉动作和运镜，不能复制其音轨。')
        if _positive_video_audio_requirement(text):
            report.errors.append('禁止引用参考视频的原声、音色、歌词、对白或声轨。')
        if _overlong_video_claim(text, material):
            report.errors.append('源视频长于本段生成时长，不能承诺完整使用超出本段的源视频或全部源帧。')
    if 'fully_copy' in text:
        report.errors.append('未核验完整音轨时长与同步关系，禁止 fully_copy。')
    if 'audio reuse' in sections['summary']:
        report.errors.append('此流程只允许音频参考，不能声称在模型内复制原音轨。')
    for name in ('overall_soundscape', 'non_diegetic_music'):
        if sections[name] != 'N/A':
            report.errors.append(name+' 必须仅为 N/A；原音频由后续流程恢复。')
    for name in REF_FIELDS[:4]:
        if not re.search(r'[\u3400-\u9fff]', sections[name]):
            report.errors.append(name+' 的说明正文必须使用中文，固定标签保留原格式。')
    audio = set(re.findall(r'<Audio \d+>', text))
    if audio:
        if audio != {'<Audio 1>'}:
            report.errors.append('本插件至多引用实际输入槽 <Audio 1>。')
        if 'audio reference' not in sections['summary']:
            report.errors.append('Audio 参考必须在 summary 中声明 audio reference。')
        audio_definition = next((line for line in definitions.splitlines() if line.startswith('<Audio 1>')), '')
        if not re.search(r'<Subject [1-9]\d*>', audio_definition) or '(S1)' not in audio_definition:
            report.errors.append('<Audio 1> 必须绑定已声明 Subject 及其全局说话人 (S1)。')
        if re.search(r'(?m)^<Audio 1>[^\n]*[:：]\s*(?!reference\s*-|weak_reference\s*-)[a-z_]+\s*-', sections['retention_analysis']):
            report.errors.append('Audio 只能声明 reference 或 weak_reference，不能声称复制音轨。')
    if material is not None:
        kind = material['visual_type']
        slot = material.get('audio_reference_slot')
        if audio and (material.get('audio_reference_available') is not True or type(slot) is not int or slot != 1):
            report.errors.append('当前工作流没有已核实的 Audio 1 输入映射，禁止声明 Audio。')
        if kind in ('atmosphere', 'environment'):
            if audio or re.search(r'\(S\d+(?:,S\d+)*\)|</?d>|\baudio (?:reference|reuse)\b', text):
                report.errors.append(kind+' 画面禁止 Audio、说话人和对白标签。')
            if _positive_vocal_requirement(text):
                report.errors.append(kind+' 画面禁止正向演唱、说话或口型要求。')
    if report.errors:
        raise ValueError('扩写结果不符合 MiniMax H3 Ref2VA 规则：'+'；'.join(report.errors[:8]))
    return text


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _field_positions(text: str, fields: tuple[str, ...]) -> dict[str, list[int]]:
    return {
        name: [match.start() for match in re.finditer(rf"(?m)^{re.escape(name)}:", text)]
        for name in fields
    }


def _check_fields(text: str, required: tuple[str, ...], report: Report) -> None:
    positions = _field_positions(text, required)
    for name, found in positions.items():
        if not found:
            report.errors.append(f"Missing required field: {name}")
        elif len(found) > 1:
            report.errors.append(f"Field appears more than once: {name}")
    if all(len(positions[name]) == 1 for name in required):
        ordered = [positions[name][0] for name in required]
        if ordered != sorted(ordered):
            report.errors.append("Required fields are not in the prescribed order")
        for name in required[1:]:
            prefix = text[: positions[name][0]]
            if not re.search(r"\r?\n[ \t]*\r?\n[ \t]*$", prefix):
                report.errors.append(f"Required field must be preceded by one blank line: {name}")


def _check_prompt_start(text: str, mode: str, report: Report) -> None:
    content = text.lstrip("\ufeff\r\n\t ")
    if mode == "t2va" and not content.startswith("integrated_multimodal_description:"):
        report.errors.append("T2VA must begin directly with integrated_multimodal_description")
    elif mode == "ref2va" and not content.startswith("subject_definitions:"):
        report.errors.append("Ref2VA must begin directly with subject_definitions")


def _check_dialogue(text: str, timeline: str, report: Report) -> None:
    opening_count = text.count("<d>")
    closing_count = text.count("</d>")
    if opening_count != closing_count:
        report.errors.append(
            f"Dialogue tags are unbalanced: {opening_count} opening and {closing_count} closing"
        )

    blocks = re.findall(r"<d>(.*?)</d>", text, flags=re.DOTALL)
    for block in blocks:
        if not re.match(r"^\[[^\]\r\n]+\]\s+\S", block.strip(), flags=re.DOTALL):
            report.errors.append(
                "Every <d> block must start with [Language] followed by spoken or sung content"
            )

    outside_timeline = text.replace(timeline, "", 1) if timeline else text
    if "<d>" in outside_timeline or "</d>" in outside_timeline:
        report.errors.append("Dialogue and lyrics must appear only in the timeline field")


def _parse_seconds(minutes: str, seconds: str) -> float:
    return int(minutes) * 60 + float(seconds)


def _check_shots(text: str, duration: float, report: Report) -> None:
    shots = list(SHOT_RE.finditer(text))
    if not shots:
        report.errors.append("No [Shot N] markers found")
        return

    numbers = [int(match.group(1)) for match in shots]
    expected = list(range(1, len(numbers) + 1))
    if numbers != expected:
        report.errors.append(f"Shot numbers must be sequential from 1; found {numbers}")

    if shots[0].group(2) is not None:
        report.errors.append("[Shot 1] must not have a timestamp")

    previous = 0.0
    for match in shots[1:]:
        if match.group(2) is None:
            report.errors.append(f"[Shot {match.group(1)}] is missing 'At MM:SS.mmm,'")
            continue
        current = _parse_seconds(match.group(2), match.group(3))
        if current <= previous:
            report.errors.append("Later shot timestamps must be strictly increasing")
        if current >= duration:
            report.errors.append(
                f"Shot {match.group(1)} starts at {current:.3f}s, outside a {duration:.2f}s video"
            )
        previous = current


def _check_placeholders(text: str, report: Report) -> None:
    placeholders = re.findall(r"\{[^{}\n]+\}", text)
    if placeholders:
        report.errors.append(f"Unresolved placeholders: {', '.join(sorted(set(placeholders)))}")
    for token in ("S.SS", "Shot N", "TODO"):
        if token in text:
            report.errors.append(f"Unresolved template token: {token}")
    if "..." in text:
        report.warnings.append("Prompt still contains an ellipsis placeholder or vague omission ('...')")


def _check_language(text: str, report: Report) -> None:
    scrubbed = re.sub(r"<d>.*?</d>", "", text, flags=re.DOTALL)
    scrubbed = re.sub(r'"[^"\n]*"', "", scrubbed)
    scrubbed = re.sub(
        r"(?m)^(?:integrated_multimodal_description|overall_soundscape|"
        r"non_diegetic_music|subject_definitions|summary|retention_analysis|"
        r"detailed_description):",
        "",
        scrubbed,
    )
    scrubbed = re.sub(r"<(?:Subject|Picture|Video|Audio) \d+>", "", scrubbed)
    scrubbed = re.sub(r"\[Shot \d+\](?: At \d{2}:\d{2}\.\d{3},)?", "", scrubbed)
    scrubbed = re.sub(r"\[(?:Chinese|English|Japanese|Korean|French|German|Italian|Spanish|Portuguese|Russian|Arabic)\]", "", scrubbed)
    for token in sorted(TASK_TYPES | VISIBLE_MARKERS | AUDIO_MARKERS, key=len, reverse=True):
        scrubbed = scrubbed.replace(token, "")
    scrubbed = re.sub(
        r"(?m)^(?:For the target video, at 0\.00 seconds into the target video,.*|"
        r"How the reference pictures align with the target video —.*)$",
        "",
        scrubbed,
    )
    scrubbed = scrubbed.replace(
        "The target video is an edited version of <Video 1>.", ""
    )
    scrubbed = scrubbed.replace("says in an off-screen voiceover", "")
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", scrubbed))
    latin_count = len(re.findall(r"[A-Za-z]", scrubbed))
    if latin_count > 60 and latin_count > max(60, cjk_count * 3):
        report.errors.append(
            "Model-facing descriptive prose appears substantially English; the local H3 contract requires Chinese prose while preserving fixed English tokens"
        )


def _check_alignment(
    text: str, timeline: str, mode: str, duration: float, report: Report
) -> None:
    content = text.lstrip("\ufeff\n\r\t ")
    first_line = content.splitlines()[0] if content else ""
    expected_time = f"{duration:.2f}"

    if mode == "t2va":
        if first_line.startswith("For the target video") or first_line.startswith(
            "How the reference pictures align"
        ):
            report.errors.append("T2VA must not start with an image-alignment instruction")
        return

    first_break = content.find("\n")
    remainder = content[first_break + 1 :] if first_break >= 0 else ""
    if not re.match(
        r"^[ \t]*\r?\n[ \t]*integrated_multimodal_description:", remainder
    ):
        report.errors.append(
            "Image-alignment instruction must be followed by one blank line and integrated_multimodal_description"
        )

    if mode == "i2va":
        expected = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced."
        )
        if first_line != expected:
            report.errors.append("I2VA first-frame instruction does not match the required exact form")
        return

    if mode == "fl2va":
        pattern = re.compile(
            r"^How the reference pictures align with the target video — "
            r"Picture 1 \(from Shot 1\) aligns with the 0\.00-second mark of the target video; "
            r"Picture 2 \(from Shot (\d+)\) aligns with the (\d+\.\d{2})-second mark of the target video\.$"
        )
        match = pattern.match(first_line)
        if not match:
            report.errors.append("FL2VA alignment instruction does not match the required exact form")
        else:
            if match.group(2) != expected_time:
                report.errors.append(
                    f"FL2VA end alignment must be {expected_time} seconds, found {match.group(2)}"
                )
            shot_numbers = [int(item.group(1)) for item in SHOT_RE.finditer(timeline)]
            if shot_numbers and int(match.group(1)) != shot_numbers[-1]:
                report.errors.append("FL2VA Picture 2 must refer to the actual final shot")
        return

    if mode == "l2va":
        pattern = re.compile(
            r"^How the reference pictures align with the target video — "
            r"<Picture 1> \(from \[Shot (\d+)\]\) aligns with the (\d+\.\d{2})-second mark of the target video\.$"
        )
        match = pattern.match(first_line)
        if not match:
            report.errors.append("L2VA alignment instruction does not match the required exact form")
        else:
            if match.group(2) != expected_time:
                report.errors.append(
                    f"L2VA end alignment must be {expected_time} seconds, found {match.group(2)}"
                )
            shot_numbers = [int(item.group(1)) for item in SHOT_RE.finditer(timeline)]
            if shot_numbers and int(match.group(1)) != shot_numbers[-1]:
                report.errors.append("L2VA Picture 1 must refer to the actual final shot")


def _extract_section(text: str, field: str, ordered_fields: tuple[str, ...]) -> str:
    start_match = re.search(rf"(?m)^{re.escape(field)}:[ \t]*", text)
    if not start_match:
        return ""
    start = start_match.end()
    later = []
    for candidate in ordered_fields:
        match = re.search(rf"(?m)^{re.escape(candidate)}:", text[start:])
        if match:
            later.append(start + match.start())
    end = min(later) if later else len(text)
    return text[start:end].strip()


def _check_ref2va(text: str, report: Report) -> None:
    if re.search(r"(?m)^integrated_multimodal_description:", text):
        report.errors.append("Ref2VA must use detailed_description, not integrated_multimodal_description")
    first_line = text.lstrip("\ufeff\n\r\t ").splitlines()[0] if text.strip() else ""
    if first_line.startswith("For the target video") or first_line.startswith(
        "How the reference pictures align"
    ):
        report.errors.append("Ref2VA must not use a Base-mode image-alignment instruction")

    definitions = _extract_section(text, "subject_definitions", REF_FIELDS)
    summary = _extract_section(text, "summary", REF_FIELDS)
    retention = _extract_section(text, "retention_analysis", REF_FIELDS)
    details = _extract_section(text, "detailed_description", REF_FIELDS)
    soundscape = _extract_section(text, "overall_soundscape", REF_FIELDS)
    music = _extract_section(text, "non_diegetic_music", REF_FIELDS)

    tracked: list[tuple[str, str, int]] = []
    for line in (item.strip() for item in definitions.splitlines() if item.strip()):
        match = REF_DEFINITION_RE.match(line)
        if not match:
            report.errors.append(
                f"Every subject_definitions item must be one line beginning '<Type N> 是' or '<Type N>：': {line}"
            )
            continue
        tracked.append(
            (match.group(1), match.group("kind"), int(match.group("number")))
        )

    tracked_labels = [item[0] for item in tracked]
    tracked_set = set(tracked_labels)
    if not tracked:
        report.errors.append("subject_definitions contains no tracked H3 reference labels")
    if len(tracked_labels) != len(tracked_set):
        duplicates = sorted(
            {label for label in tracked_labels if tracked_labels.count(label) > 1}
        )
        report.errors.append(f"Reference labels are defined more than once: {', '.join(duplicates)}")

    subject_numbers = [number for _, kind, number in tracked if kind == "Subject"]
    if subject_numbers and subject_numbers != list(range(1, len(subject_numbers) + 1)):
        report.errors.append(
            f"Subject labels must be sequential from 1 in definition order; found {subject_numbers}"
        )

    # Material numbers identify actual input slots, including inline provenance.
    # Their existence and mapping cannot be verified from prompt text alone.
    material_labels = set(re.findall(r"<(?:Picture|Video|Audio) [^<>\s]+>", text))
    for label in sorted(material_labels):
        number = label.rsplit(" ", 1)[1][:-1]
        if not number.isdecimal() or int(number) < 1:
            report.errors.append(
                f"Material label indices must be positive integers: {label}"
            )

    outside_definitions = "\n".join((summary, retention, details, soundscape, music))
    outside_labels = set(REF_LABEL_RE.findall(outside_definitions))
    untracked = sorted(outside_labels - tracked_set)
    if untracked:
        report.errors.append(
            "Labels used outside subject_definitions without a tracked definition line: "
            + ", ".join(untracked)
        )

    task_match = re.match(r"\[([^\]]+)\]", summary)
    found_types: list[str] = []
    if not task_match:
        report.errors.append("summary must begin with a square-bracketed Ref2VA task type")
    else:
        found_types = [item.strip() for item in task_match.group(1).split("+")]
        invalid = sorted(set(found_types) - TASK_TYPES)
        if invalid:
            report.errors.append(f"Unknown Ref2VA task types: {', '.join(invalid)}")
        duplicates = sorted({item for item in found_types if found_types.count(item) > 1})
        if duplicates:
            report.errors.append(f"Ref2VA task types must not repeat: {', '.join(duplicates)}")

        kinds = {kind for _, kind, _ in tracked}
        type_requirements = {
            "keyframe completion": "Picture",
            "video editing": "Video",
            "video continuation": "Video",
            "audio reuse": "Audio",
            "audio reference": "Audio",
        }
        for task_type, required_kind in type_requirements.items():
            if task_type in found_types and required_kind not in kinds:
                report.errors.append(
                    f"Task type '{task_type}' requires a tracked <{required_kind} N> definition"
                )

        if "video editing" in found_types:
            prose = summary[task_match.end() :].lstrip()
            expected = "The target video is an edited version of <Video 1>."
            if not prose.startswith(expected):
                report.errors.append(
                    f"video editing summary must begin after its task prefix with: {expected}"
                )

    if re.search(r"\(S\d+(?:,S\d+)*\)", retention):
        report.errors.append("retention_analysis must not contain speaker IDs")

    retention_lines = [line.strip() for line in retention.splitlines() if line.strip()]
    retained_labels: list[str] = []
    for line in retention_lines:
        label_match = REF_LABEL_RE.match(line)
        if not label_match:
            report.errors.append(f"Every retention_analysis item must begin with a tracked label: {line}")
            continue
        label = label_match.group(0)
        retained_labels.append(label)
        if label not in tracked_set:
            report.errors.append(f"Retention line uses an untracked label: {label}")
        marker_match = re.search(r"[:：]\s*([a-z_]+)\s*-", line)
        if not marker_match:
            report.errors.append(f"Retention line has no fixed relationship marker: {line}")
            continue
        marker = marker_match.group(1)
        allowed = AUDIO_MARKERS if label.startswith("<Audio") else VISIBLE_MARKERS
        if marker not in allowed:
            report.errors.append(f"Invalid retention marker '{marker}' for {label}")

    duplicate_retention = sorted(
        {label for label in retained_labels if retained_labels.count(label) > 1}
    )
    if duplicate_retention:
        report.errors.append(
            f"Tracked labels have more than one retention line: {', '.join(duplicate_retention)}"
        )
    missing_retention = sorted(tracked_set - set(retained_labels))
    if missing_retention:
        report.errors.append(
            f"Tracked labels without exactly one retention_analysis line: {', '.join(missing_retention)}"
        )

    shot_one = details.find("[Shot 1]")
    if shot_one >= 0:
        style_opening = details[:shot_one].strip()
        if not style_opening:
            report.errors.append("Ref2VA detailed_description must establish style before [Shot 1]")
        elif len(re.findall(r"[.!?。！？]", style_opening)) > 2:
            report.warnings.append(
                "Ref2VA style opening before [Shot 1] should normally be one or two sentences"
            )

    cjk_units = len(re.findall(r"[\u3400-\u9fff]", details))
    latin_words = len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", details))
    information_units = cjk_units + latin_words * 2
    if "video editing" not in found_types and information_units < 500:
        report.warnings.append(
            f"detailed_description information density is low ({information_units} approximate units); verify that every shot covers composition, subjects, environment, lighting, actions, camera, sound, and reference timing"
        )
    elif "video editing" not in found_types and information_units > 1400:
        report.warnings.append(
            f"detailed_description is very dense ({information_units} approximate units); verify that the requested timeline can contain all described events"
        )


def validate_prompt(text: str, mode: str, duration: float) -> Report:
    report = Report()
    if not 4 <= duration <= 15:
        report.errors.append("Duration must be between 4 and 15 seconds")
    if not text.strip():
        report.errors.append("Prompt is empty")
        return report

    _check_placeholders(text, report)
    _check_language(text, report)
    _check_prompt_start(text, mode, report)

    if mode == "ref2va":
        _check_fields(text, REF_FIELDS, report)
        _check_ref2va(text, report)
        timeline = _extract_section(text, "detailed_description", REF_FIELDS)
    else:
        _check_fields(text, BASE_FIELDS, report)
        timeline = _extract_section(text, "integrated_multimodal_description", BASE_FIELDS)
        _check_alignment(text, timeline, mode, duration, report)
        for ref_only_field in (
            "subject_definitions",
            "summary",
            "retention_analysis",
            "detailed_description",
        ):
            if re.search(rf"(?m)^{ref_only_field}:", text):
                report.errors.append(
                    f"Base modes must not contain Ref2VA-only field: {ref_only_field}"
                )

    _check_dialogue(text, timeline, report)
    _check_shots(timeline, duration, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["t2va", "i2va", "fl2va", "l2va", "ref2va"], required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    args = parser.parse_args()

    try:
        text = args.prompt_file.read_text(encoding="utf-8-sig")
    except OSError as exc:
        print(f"ERROR: cannot read prompt file: {exc}", file=sys.stderr)
        return 2

    report = validate_prompt(text, args.mode, args.duration)
    for item in report.errors:
        print(f"ERROR: {item}")
    for item in report.warnings:
        print(f"WARNING: {item}")
    if report.ok:
        print(f"VALID: {args.mode} prompt passed with {len(report.warnings)} warning(s)")
        return 0
    print(f"INVALID: {len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
