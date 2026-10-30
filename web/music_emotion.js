export const musicEmotionUpgradeNotice = "后端尚未加载音乐情绪分析，请重启 ComfyUI 并刷新。";

export function musicEmotionBlocked({plan, projectId, generationBusy = false}) {
  if (!plan || plan.id !== projectId || !plan.segments?.length) return "请先选择有分段的分析项目。";
  if (plan.music_emotion_capabilities?.analyze !== true) return musicEmotionUpgradeNotice;
  if (plan.expansion_active) return "批量扩写进行中，请等待批次结束。";
  if (generationBusy || plan.controller_active || ["running", "pausing", "stopping", "merging"].includes(plan.run_status)) return "视频生成进行中，请等待当前任务结束。";
  return "";
}

export function musicEmotionSummary(row) {
  const emotion = row.music_emotion;
  if (!emotion) return "尚未分析音乐情绪。按当前切点试听原曲，并结合旋律、节奏、音色、演唱与歌词生成导演参考。";
  if (emotion.status === "stale") return "音乐情绪已过期，请按当前分段重新分析。";
  const value = emotion.analysis || {};
  const labels = Array.isArray(value.labels) ? value.labels.join("、") : value.labels;
  const intensity = Number(value.intensity);
  return [labels, value.intensity != null && Number.isFinite(intensity) ? `强度 ${intensity}/5（模型估计）` : "", value.trend, value.summary].filter(Boolean).join(" · ");
}

export function createMusicEmotionActions({getState, requestProject, saveDraft, refresh, changed = () => {},
  schedule = callback => setTimeout(callback, 1500), unschedule = clearTimeout}) {
  let projectId = "", version = 0, disposed = false, timer = null;
  let pending = false, cancelling = false, result = null, error = "", terminal = "";
  const current = (id, token) => !disposed && token === version && id === projectId && getState().projectId === id;
  const stopTimer = () => {if (timer !== null) unschedule(timer); timer = null;};
  const active = () => pending || result?.job?.status === "running" || Boolean(getState().plan?.music_emotion_active);
  function switchProject(id) {
    if (id === projectId && !disposed) return;
    stopTimer(); version += 1; projectId = id; disposed = false;
    pending = false; cancelling = false; result = null; error = ""; terminal = "";
    changed();
  }
  async function accept(value, id, token) {
    if (!current(id, token)) return;
    result = value; error = ""; changed();
    const job = value.job;
    if (job?.status === "running") {
      stopTimer(); timer = schedule(() => {timer = null; void poll(id, token);});
    } else if (job && terminal !== `${job.id}:${job.status}`) {
      terminal = `${job.id}:${job.status}`;
      try {await refresh(id);}
      catch (failure) {if (current(id, token)) terminal = ""; throw failure;}
    }
  }
  async function poll(id, token) {
    try {
      const value = await requestProject(id, "/music-emotion");
      if (current(id, token)) await accept(value, id, token);
    } catch (failure) {
      if (current(id, token)) {
        stopTimer(); error = `无法读取情绪分析状态：${failure.message}。请点击“刷新情绪状态”重试；后台任务可能仍在进行。`; changed();
      }
    }
  }
  return {
    get busy() {return active();},
    get cancelling() {return cancelling;},
    get result() {return result;},
    get error() {return error;},
    switchProject,
    async connect(plan) {
      switchProject(plan?.id || "");
      if (disposed || pending || !plan || plan.music_emotion_capabilities?.analyze !== true) return;
      stopTimer();
      await poll(projectId, version);
    },
    async start(indices) {
      const state = getState(), reason = musicEmotionBlocked(state);
      if (reason) throw new Error(reason);
      if (active()) throw new Error("音乐情绪分析正在进行，请等待或取消当前任务。");
      stopTimer();
      const id = state.projectId, token = ++version;
      pending = true; error = ""; changed();
      try {
        const revision = await saveDraft();
        if (!current(id, token)) return;
        const blocked = musicEmotionBlocked(getState());
        if (blocked) throw new Error(blocked);
        const value = await requestProject(id, "/music-emotion", {revision, indices});
        if (current(id, token)) await accept(value, id, token);
      } catch (failure) {
        if (current(id, token)) {error = failure.message; throw failure;}
      } finally {
        if (current(id, token)) {pending = false; changed();}
      }
    },
    async cancel() {
      if (!active() || cancelling) return;
      if (pending && result?.job?.status !== "running") return;
      const id = projectId, token = ++version;
      cancelling = true; changed(); stopTimer();
      try {
        const value = await requestProject(id, "/music-emotion/cancel", {});
        if (current(id, token)) await accept(value, id, token);
      } catch (failure) {
        if (current(id, token)) {error = `取消情绪分析失败：${failure.message}。请刷新情绪状态后重试。`; throw failure;}
      } finally {
        if (current(id, token)) {cancelling = false; changed();}
      }
    },
    async refreshStatus() {
      if (pending || cancelling) return;
      stopTimer(); await poll(projectId, version);
    },
    dispose() {stopTimer(); disposed = true; version += 1;},
  };
}
