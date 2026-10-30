import { api } from "../../scripts/api.js";
import { openAssetLibrary, mergeVideoLibraryImport, validateVideoReferences, mentionedVideoNumbers, remapVideoMentions } from "./asset_library.js";

function el(tag, parent, text) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  parent.append(item); return item;
}
export function hasVideoConfiguration(plan, row) {
  if (plan.video_materials_version === 1) return true;
  const target = row || plan;
  return row ? Object.hasOwn(target, "video_reference_source") || Object.hasOwn(target, "video_refs") || Object.hasOwn(target, "video_note") :
    Object.hasOwn(target, "default_video_refs") || Object.hasOwn(target, "default_video_note");
}
// Append only: old graph output links 0..12 keep their slot identities.
export function appendVideoOutputs(node) {
  if (!Array.isArray(node.outputs) || node.outputs.length < 13 || node.outputs.length >= 16) return;
  for (let slot = node.outputs.length; slot < 16; slot++) node.addOutput(`video_${slot-12}`, "IMAGE");
}
export function videoMaterialEditor(parent, {projectId, refs = [], note = "", metadata = {}, defaults,
  source = "default", configured = false, changed = () => {}, getRevision, readOnly = false}) {
  const box = el("section", parent); box.className = "h3lv-video-material-editor";
  el("h4", box, defaults ? "本段参考视频" : "项目默认参考视频");
  const value = [...refs], info = {...metadata};
  let customNote = note, managed = configured, uploading = false, locked = readOnly, mentionRange = null;
  const controls = el("div", box); controls.className = "h3lv-actions";
  let sourceSelect, copy;
  if (defaults) {
    const label = el("label", controls, "参考视频来源 "); sourceSelect = el("select", label);
    for (const [id, title] of [["default", "使用项目默认"], ["custom", "本段自定义"]]) el("option", sourceSelect, title).value = id;
    sourceSelect.value = source;
  }
  const list = el("div", box); list.className = "h3lv-reference-list h3lv-video-reference-list";
  const summary = el("p", box); summary.className = "h3lv-help"; summary.setAttribute("role", "status");
  const actions = el("div", box); actions.className = "h3lv-actions";
  const upload = el("input", box); upload.type = "file"; upload.accept = "video/*"; upload.multiple = true; upload.hidden = true;
  const button = (text, callback) => {const item = el("button", actions, text); item.type = "button"; item.className = "h3lv-button"; item.onclick = callback; return item;};
  const add = button("＋ 上传参考视频", () => {if (editable()) upload.click();});
  const library = button("从视频素材库选择", async () => {
    if (!editable()) return;
    try {
      const total = validateVideoReferences(value, info);
      await openAssetLibrary({projectId, getRevision, mediaType:"video", currentCount:value.length, currentDuration:total,
        canImport:() => box.isConnected && editable(), onImport:async (imported, choices) => {
          if (!box.isConnected || !editable()) throw new Error("编辑器已关闭或只读，请重新打开素材库。");
          const merged = mergeVideoLibraryImport(value, customNote, info, imported, choices);
          value.splice(0, value.length, ...merged.refs); customNote = merged.note; Object.assign(info, merged.metadata); mark();
        }});
    } catch (error) {window.alert(error.message);}
  });
  const mention = button("@ 引用视频", () => {textarea.focus(); openPicker(false);});
  const textarea = el("textarea", box); textarea.className = "h3lv-material-note";
  textarea.placeholder = "输入 @ 选择视频，例如：@视频1 参考运镜，@视频2 参考人物动作。";
  const picker = el("div", box); picker.className = "h3lv-mention-picker"; picker.hidden = true;
  el("p", box, "最多 3 段，每段 2–15 秒，合计不超过 15 秒。仅参考运镜和动作，不使用参考视频原声；声音继续使用项目音频。此处显示原文件时长，H3 实际参考帧按生成流程抽取。").className = "h3lv-help";
  const inherited = () => Boolean(defaults && sourceSelect.value === "default");
  const editable = () => !locked && !uploading && !inherited();
  const current = () => inherited() ? defaults() : {refs:value, note:customNote, metadata:info};
  const mark = () => {managed = true; render(); changed();};
  function closePicker() {picker.hidden = true; mentionRange = null;}
  function openPicker(typed) {
    if (!editable() || !value.length) return;
    const start = textarea.selectionStart || 0;
    mentionRange = {start:typed ? Math.max(0,start-1) : start, end:textarea.selectionEnd || start};
    picker.replaceChildren();
    value.forEach((name, position) => {
      const choice = el("button", picker, `@视频${position+1}`); choice.type = "button"; choice.className = "h3lv-mention-choice";
      choice.onclick = () => {
        if (!editable()) return;
        const {start,end} = mentionRange; const token = `@视频${position+1}`;
        customNote = customNote.slice(0,start)+token+customNote.slice(end); mark(); textarea.focus(); textarea.setSelectionRange(start+token.length,start+token.length);
      };
    });
    picker.hidden = false;
  }
  textarea.oninput = () => {
    if (!editable()) return;
    customNote = textarea.value; managed = true; changed();
    if (textarea.value[(textarea.selectionStart || 0)-1] === "@") openPicker(true); else closePicker();
  };
  textarea.onkeydown = event => {if (event.key === "Escape") closePicker();};
  function render() {
    const data = current(), linked = inherited(), disabled = !editable();
    textarea.value = data.note || ""; textarea.disabled = disabled;
    add.disabled = disabled || data.refs.length >= 3; library.disabled = disabled || !getRevision;
    upload.disabled = disabled; mention.disabled = disabled || !data.refs.length;
    if (sourceSelect) sourceSelect.disabled = locked || uploading;
    if (copy) copy.disabled = locked || uploading || !linked || !defaults().refs.length;
    closePicker(); list.replaceChildren();
    try {summary.textContent = `${data.refs.length} / 3 段 · 原视频合计 ${validateVideoReferences(data.refs, data.metadata || {}).toFixed(2)} / 15 秒`;}
    catch (error) {summary.textContent = error.message;}
    if (!data.refs.length) el("p", list, linked ? "项目默认参考视频尚未添加。" : "可选：添加用于参考运镜或动作的视频。");
    data.refs.forEach((name, position) => {
      const item = el("figure", list); item.className = "h3lv-reference-item h3lv-video-reference-item";
      const preview = el("video", item); preview.controls = true; preview.muted = true; preview.defaultMuted = true; preview.preload = "metadata"; preview.playsInline = true;
      preview.src = api.apiURL(`/h3lv/project/${encodeURIComponent(projectId)}/video-refs/${encodeURIComponent(name)}`);
      const duration = Number(data.metadata?.[name]?.duration);
      el("figcaption", item, `视频${position+1} · ${Number.isFinite(duration) ? duration.toFixed(2)+" 秒" : "时长读取中"}`);
      if (linked) return;
      const tools = el("div", item); tools.className = "h3lv-reference-tools";
      for (const [title,step] of [["上移",-1],["下移",1],["移除",0]]) {
        const b = el("button", tools, title); b.type = "button"; b.className = "h3lv-reference-tool";
        b.disabled = disabled || (step !== 0 && (position+step < 0 || position+step >= value.length));
        b.onclick = () => {
          if (!editable()) return;
          const before = [...value];
          if (!step) {
            if (mentionedVideoNumbers(customNote).has(position+1)) {window.alert(`说明正在引用 @视频${position+1}，请先删除引用再移除视频。`); return;}
            value.splice(position,1);
          } else [value[position],value[position+step]] = [value[position+step],value[position]];
          customNote = remapVideoMentions(customNote,before,value); mark();
        };
      }
    });
  }
  if (sourceSelect) {
    sourceSelect.onchange = () => {if (!locked && !uploading) mark();};
    copy = button("复制默认视频到本段", () => {
      if (locked || uploading || !inherited() || !defaults().refs.length) return;
      const data = defaults(); value.splice(0,value.length,...data.refs); customNote = data.note; Object.assign(info,data.metadata);
      sourceSelect.value = "custom"; mark();
    });
  }
  upload.onchange = async () => {
    if (!editable()) return;
    const files = [...upload.files]; upload.value = "";
    if (value.length+files.length > 3) {window.alert("参考视频最多 3 段，请减少所选文件。"); return;}
    uploading = true; render();
    try {
      const pending = [], nextInfo = {...info};
      for (const file of files) {
        const form = new FormData(); form.append("video",file,file.name);
        const response = await api.fetchApi(`/h3lv/project/${encodeURIComponent(projectId)}/video-refs`,{method:"POST",body:form});
        const result = await response.json(); if (!response.ok) throw new Error(result.error || "视频上传失败");
        pending.push(result.name); nextInfo[result.name] = result.metadata;
      }
      const merged = [...new Set([...value,...pending])]; validateVideoReferences(merged,nextInfo);
      if (locked || !box.isConnected) throw new Error("编辑器已关闭或只读，请重新添加视频。");
      value.splice(0,value.length,...merged); Object.assign(info,nextInfo); if (files.length) mark();
    } catch (error) {window.alert(error.message);}
    finally {uploading = false; render();}
  };
  const ready = Promise.all(value.filter(name => !info[name]).map(async name => {
    const response = await api.fetchApi(`/h3lv/project/${encodeURIComponent(projectId)}/video-refs/${encodeURIComponent(name)}/info`);
    const result = await response.json(); if (!response.ok) throw new Error(result.error || "无法读取视频时长"); info[name] = result;
  })).then(() => {render();}).catch(error => {summary.textContent = error.message;});
  render();
  function payload(includeSource) {
    if (!managed) return {};
    if (uploading) throw new Error("参考视频正在上传，请完成后再保存。");
    const data = current(); validateVideoReferences(data.refs, data.metadata || {});
    for (const number of mentionedVideoNumbers(data.note)) {
      if (number < 1 || number > data.refs.length) throw new Error(`视频说明引用 @视频${number}，但当前没有对应视频。`);
    }
    return {video_refs:[...value], video_note:customNote, ...(includeSource && sourceSelect ? {video_reference_source:sourceSelect.value} : {})};
  }
  return {refs:value, source:sourceSelect, note:textarea, ready, render, setReadOnly(value) {locked = Boolean(value); render();},
    getState:() => ({refs:value,note:customNote,metadata:info,configured:managed}),
    getPayload:() => payload(true),
    getDefaultPayload:() => payload(false)};
}
