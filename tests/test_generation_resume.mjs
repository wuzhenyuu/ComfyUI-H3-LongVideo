import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(new URL("../web/h3lv.js", import.meta.url), "utf8");
const start = source.indexOf("const startingProjects = new Set();");
const end = source.indexOf("async function startApprovedSequence(", start);
const plain = value => JSON.parse(JSON.stringify(value));
function deferred() {
  let resolve;
  const promise = new Promise(done => {resolve = done;});
  return {promise, resolve};
}
function project(overrides = {}) {
  return {id: "project-a", revision: 7, approved: true, run_status: "paused", controller_active: false,
    generation_capabilities: {resume_unfinished: true, single_segment: true},
    segments: [
      {index: 0, start: 0, end: 5, job: {status: "completed", video: "old-a.mp4"}, needs_regeneration: true},
      {index: 1, start: 5, end: 10, job: {status: "failed"}},
      {index: 2, start: 10, end: 15},
    ], ...overrides};
}
function harness(overrides = {}) {
  const options = [], calls = [], confirmations = [], notices = [];
  const context = {
    needsReplacement: row => Boolean(row.needs_regeneration && row.job?.status === "completed" && row.job?.video),
    element(tag, text, parent) {
      const item = {tag, text, value: ""};
      parent?.children.push(item); options.push(item); return item;
    },
  };
  vm.runInNewContext(`${source.slice(start, end)}\nglobalThis.actionsFactory = createReviewGenerationActions;
    globalThis.blocked = reviewGenerationBlocked; globalThis.syncOptions = syncGenerationSegmentOptions;
    globalThis.existingVideo = existingGenerationVideo; globalThis.retentionText = generationRetentionText;`, context);
  const state = {plan: project(), projectId: "project-a", dirty: false};
  let fresh = null, refreshed = 0, changed = 0;
  const payload = {prompt: {"1": {class_type: "H3LVUnified"}}, workflow: {nodes: []}, loader_id: "1", video_id: "4", client_id: "test"};
  const actions = context.actionsFactory({
    getState: () => state,
    async requestProject(id, suffix, body) {
      calls.push({id, suffix, body});
      if (!suffix) return fresh || structuredClone(state.plan);
      return {};
    },
    async buildPayload() {return payload;},
    async confirm(copy) {confirmations.push(copy); return true;},
    async refresh() {refreshed += 1;}, changed() {changed += 1;},
    notify(...args) {notices.push(args);}, ...overrides,
  });
  return {actions, state, calls, confirmations, notices, context, options, payload,
    setFresh(value) {fresh = value;}, get refreshed() {return refreshed;}, get changed() {return changed;}};
}

test("resume posts the current graph and approved revision once, without retry or preview clearing", async () => {
  const h = harness();
  const oldSegments = structuredClone(h.state.plan.segments);
  await h.actions.resume();
  assert.deepEqual(plain(h.calls), [
    {id: "project-a"},
    {id: "project-a", suffix: "/run", body: {...h.payload, resume_unfinished: true, revision: 7}},
  ]);
  assert.deepEqual(h.state.plan.segments, oldSegments);
  assert.equal(h.refreshed, 1);
  assert.equal(h.actions.busy, false);
  assert.match(h.notices[0][1], /待重生成的旧结果/);
});

test("selected generation submits only its numeric index and revision to regenerate", async () => {
  const h = harness();
  await h.actions.single("2");
  assert.deepEqual(plain(h.calls[1]), {id: "project-a", suffix: "/regenerate", body: {...h.payload, index: 2, revision: 7}});
  assert.equal(h.calls.length, 2);
  assert.match(h.confirmations[0].message, /完成后停止/);
  assert.equal(h.calls.some(call => call.suffix === "/run" || call.suffix === "/retry"), false);
});

test("replacing a completed selected segment confirms retaining its video until success", async () => {
  const h = harness();
  await h.actions.single("0");
  assert.match(h.confirmations[0].message, /旧视频保留到新视频成功生成/);
  assert.equal(h.state.plan.segments[0].job.video, "old-a.mp4");
});

test("a completed replacement previous video is counted and confirmed as retained before selected regeneration", async () => {
  const h = harness();
  h.state.plan.segments[1].replacement_previous_job = {status: "completed", video: "previous-b.mp4"};
  await h.actions.single("1");
  assert.match(h.confirmations[0].message, /本段旧视频保留到新视频成功生成/);
  assert.equal(h.state.plan.segments[1].replacement_previous_job.video, "previous-b.mp4");
  assert.equal(h.context.retentionText(h.state.plan), "已有 2 段视频可保留（含 1 段待恢复旧片）");
});

test("resume retains and reports completed previous results even when the current job is missing", async () => {
  const h = harness();
  h.state.plan.segments[2].replacement_previous_job = {status: "completed", video: "previous-c.mp4"};
  const segments = structuredClone(h.state.plan.segments);
  const select = {value: "", children: [], replaceChildren() {this.children = [];}};
  h.context.syncOptions(select, h.state.plan, 2, null);
  assert.match(select.children[2].text, /已有视频（待恢复）/);
  assert.doesNotMatch(select.children[2].text, /待生成/);
  await h.actions.resume();
  assert.match(h.notices[0][1], /已有 2 段视频可保留（含 1 段待恢复旧片）/);
  assert.deepEqual(h.state.plan.segments, segments);
});

test("completed current videos take priority and incomplete previous results are not counted", () => {
  const h = harness();
  const current = h.state.plan.segments[0];
  current.replacement_previous_job = {status: "completed", video: "previous-a.mp4"};
  assert.equal(h.context.existingVideo(current), current.job);
  assert.equal(h.context.retentionText(h.state.plan), "已有 1 段视频可保留");
  for (const previous of [{status: "failed", video: "failed.mp4"}, {status: "completed"}, {}]) {
    h.state.plan.segments[1].replacement_previous_job = previous;
    assert.equal(h.context.existingVideo(h.state.plan.segments[1]), null);
    assert.equal(h.context.retentionText(h.state.plan), "已有 1 段视频可保留");
  }
});

test("old backend capabilities block both actions locally and when absent on fresh read", async () => {
  for (const action of ["resume", "single"]) {
    const local = harness();
    delete local.state.plan.generation_capabilities;
    await assert.rejects(local.actions[action]("1"), /后端尚未加载此功能，请重启ComfyUI并刷新/);
    assert.equal(local.calls.length, 0);
    const changed = harness();
    changed.setFresh(project({generation_capabilities: undefined}));
    await assert.rejects(changed.actions[action]("1"), /后端尚未加载此功能/);
    assert.equal(changed.calls.length, 1);
  }
});

test("dirty, unapproved, expanding, active, and empty projects cannot submit", async () => {
  for (const change of [
    state => {state.dirty = true;},
    state => {state.plan.approved = false;},
    state => {state.plan.expansion_active = true;},
    state => {state.plan.controller_active = true;},
    state => {state.plan.run_status = "merging";},
    state => {state.plan.segments = [];},
    state => {state.plan = null;},
    state => {state.projectId = "";},
  ]) {
    for (const action of ["resume", "single"]) {
      const h = harness(); change(h.state);
      await assert.rejects(h.actions[action]("1"));
      assert.equal(h.calls.length, 0);
    }
  }
});

test("inactive persisted run statuses allow restart recovery, actual controllers remain blocked", async () => {
  for (const run_status of ["running", "pausing", "stopping"]) {
    const h = harness(); h.state.plan.run_status = run_status;
    await h.actions.resume();
    assert.equal(h.calls[1].body.resume_unfinished, true);
    const unknown = harness(); unknown.state.plan.run_status = run_status;
    delete unknown.state.plan.controller_active;
    await assert.rejects(unknown.actions.resume(), /生成任务正在进行/);
  }
});

test("invalid selected indexes never request or build generation", async () => {
  for (const index of [undefined, null, "", " ", "-1", "3", "0.5", "bad", Infinity]) {
    const h = harness({buildPayload() {throw new Error("must not build");}});
    await assert.rejects(h.actions.single(index), /请选择有效的分段/);
    assert.equal(h.calls.length, 0);
  }
});

test("cancelled selection does not build or post and releases the project lock", async () => {
  const h = harness({async confirm() {return false;}, buildPayload() {throw new Error("must not build");}});
  await h.actions.single("0");
  assert.equal(h.calls.length, 0);
  assert.equal(h.actions.busy, false);
  assert.equal(h.changed, 2);
  await h.actions.resume().catch(error => assert.match(error.message, /must not build/));
});

test("duplicate clicks and the other new action cannot submit while building payload", async () => {
  const gate = deferred();
  const h = harness({buildPayload: () => gate.promise});
  const pending = h.actions.resume();
  assert.equal(h.actions.busy, true);
  await assert.rejects(h.actions.resume(), /重复提交/);
  await assert.rejects(h.actions.single("1"), /重复提交/);
  gate.resolve(h.payload);
  await pending;
  assert.equal(h.calls.filter(call => call.suffix).length, 1);
});

test("project switching, edits, and revision changes during payload selection stop submission", async () => {
  for (const mutate of [
    state => {state.projectId = "project-b"; state.plan = project({id: "project-b"});},
    state => {state.dirty = true;},
    state => {state.plan.revision += 1;},
  ]) {
    const gate = deferred();
    const h = harness({buildPayload: () => gate.promise});
    const pending = h.actions.single("1");
    await Promise.resolve();
    mutate(h.state); gate.resolve(h.payload);
    await assert.rejects(pending);
    assert.equal(h.calls.length, 0);
  }
});

test("fresh revision, approval, or activity changes cannot silently overwrite another window", async () => {
  for (const overrides of [{revision: 8}, {approved: false}, {controller_active: true}, {expansion_active: true}]) {
    const h = harness(); h.setFresh(project(overrides));
    await assert.rejects(h.actions.resume());
    assert.equal(h.calls.length, 1);
    assert.equal(h.refreshed, 0);
  }
});

test("a late project switch while reading latest cannot post to either project", async () => {
  const gate = deferred();
  let posts = 0;
  const h = harness({async requestProject(id, suffix) {if (suffix) {posts += 1; return {};} return gate.promise;}});
  const pending = h.actions.resume();
  await Promise.resolve();
  h.state.projectId = "project-b"; h.state.plan = project({id: "project-b"});
  gate.resolve(project());
  await assert.rejects(pending, /分析项目或版本已改变/);
  assert.equal(posts, 0);
});

test("poll option rebuild preserves user choice; new projects default to audition or first segment", () => {
  const h = harness();
  const select = {value: "", children: [], replaceChildren() {this.children = [];}};
  h.context.syncOptions(select, h.state.plan, 1, null);
  assert.equal(select.value, "1");
  assert.match(select.children[0].text, /第 1 段.*0.000—5.000s.*保留旧片/);
  select.value = "2";
  h.context.syncOptions(select, project(), 0, select.value);
  assert.equal(select.value, "2");
  h.context.syncOptions(select, project({id: "new"}), 0, null);
  assert.equal(select.value, "0");
  h.context.syncOptions(select, project({segments: []}), 0, select.value);
  assert.equal(select.value, "");
  h.context.syncOptions(select, project(), -1, null);
  assert.equal(select.value, "0");
});
