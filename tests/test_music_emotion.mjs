import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

const source = fs.readFileSync(new URL("../web/music_emotion.js", import.meta.url), "utf8");
const {createMusicEmotionActions, musicEmotionBlocked, musicEmotionSummary} = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
const project = (id = "project-a") => ({id, revision:7, run_status:"draft", music_emotion_capabilities:{analyze:true},
  music_emotion_active:false, segments:[{index:0}, {index:1}]});
const deferred = () => {let resolve; const promise = new Promise(done => {resolve = done;}); return {promise, resolve};};
const running = (done = 0) => ({ready:true, job:{id:"job-a", status:"running", done, total:2, index:done}});
const finished = status => ({ready:true, job:{id:"job-a", status, done:2, total:2}});

function harness(overrides = {}) {
  const state = {plan:project(), projectId:"project-a"}, calls = [], timers = new Map();
  let nextTimer = 0, refreshed = 0, changes = 0;
  const actions = createMusicEmotionActions({
    getState:() => state,
    requestProject:async (id, suffix, body) => {calls.push({id,suffix,body}); return body ? running() : {ready:true,job:null};},
    saveDraft:async () => {calls.push({save:true}); state.plan.revision = 8; return 8;},
    refresh:async () => {refreshed += 1; state.plan.music_emotion_active = false;},
    changed:() => {changes += 1;},
    schedule:callback => {const id = ++nextTimer; timers.set(id,callback); return id;},
    unschedule:id => timers.delete(id), ...overrides,
  });
  actions.switchProject(state.projectId);
  return {state, actions, calls, timers, get refreshed() {return refreshed;}, get changes() {return changes;},
    async tick() {const [id,callback] = timers.entries().next().value; timers.delete(id); await callback(); await new Promise(resolve => setImmediate(resolve));}};
}

test("old services disable analysis without requesting an absent endpoint", async () => {
  const h = harness(); delete h.state.plan.music_emotion_capabilities;
  assert.match(musicEmotionBlocked(h.state), /重启 ComfyUI/);
  await h.actions.connect(h.state.plan);
  await assert.rejects(h.actions.start([0]), /重启 ComfyUI/);
  assert.equal(h.calls.length,0); assert.equal(h.timers.size,0);
});

test("analysis saves current cuts and briefs before submitting the resulting revision", async () => {
  const h = harness();
  await h.actions.start([1]);
  assert.deepEqual(h.calls,[{save:true},{id:"project-a",suffix:"/music-emotion",body:{revision:8,indices:[1]}}]);
  assert.equal(h.actions.busy,true); assert.equal(h.timers.size,1);
});

test("save failure prevents submission and restores editable state", async () => {
  const h = harness({saveDraft:async () => {throw new Error("草稿版本冲突");}});
  await assert.rejects(h.actions.start([0]),/草稿版本冲突/);
  assert.equal(h.calls.length,0); assert.equal(h.actions.busy,false);
  assert.equal(h.actions.error,"草稿版本冲突");
});

test("polling completion refreshes director briefs once and stops polling", async () => {
  const calls = [];
  const h = harness({requestProject:async (id,suffix,body) => {calls.push({id,suffix,body}); return body ? running() : finished("completed");}});
  await h.actions.start([0,1]); await h.tick();
  assert.equal(h.refreshed,1); assert.equal(h.timers.size,0); assert.equal(h.actions.busy,false);
  await h.actions.refreshStatus(); assert.equal(h.refreshed,1);
  assert.equal(calls.length,3);
});

test("opening a project reconnects to its running background job", async () => {
  const h = harness({requestProject:async () => running(1)});
  h.state.plan.music_emotion_active = true;
  await h.actions.connect(h.state.plan);
  assert.equal(h.actions.result.job.done,1); assert.equal(h.actions.busy,true); assert.equal(h.timers.size,1);
});

test("switching projects discards an in-flight status response and stops its poll", async () => {
  const response = deferred();
  const h = harness({requestProject:() => response.promise});
  const connected = h.actions.connect(h.state.plan);
  h.state.projectId = "project-b"; h.state.plan = project("project-b"); h.actions.switchProject("project-b");
  const changes = h.changes;
  response.resolve(finished("completed")); await connected;
  assert.equal(h.actions.result,null); assert.equal(h.refreshed,0); assert.equal(h.changes,changes); assert.equal(h.timers.size,0);
});

test("closing stops polling and ignores responses without cancelling background work", async () => {
  const response = deferred(), calls = [];
  const h = harness({requestProject:(id,suffix,body) => {calls.push(suffix); return body ? Promise.resolve(running()) : response.promise;}});
  await h.actions.start([0]);
  const polling = h.actions.refreshStatus(); h.actions.dispose(); response.resolve(finished("completed")); await polling;
  assert.equal(h.timers.size,0); assert.equal(h.refreshed,0);
  assert.equal(calls.includes("/music-emotion/cancel"),false);
});

test("switching while draft saves prevents submission for either project", async () => {
  const saved = deferred(); const h = harness({saveDraft:() => saved.promise});
  const started = h.actions.start([0]);
  h.state.projectId = "project-b"; h.state.plan = project("project-b"); h.actions.switchProject("project-b");
  saved.resolve(8); await started;
  assert.equal(h.calls.length,0); assert.equal(h.actions.busy,false);
});

test("cancel targets only the emotion job and refreshes the unlocked project", async () => {
  const calls = [];
  const h = harness({requestProject:async (id,suffix,body) => {calls.push({id,suffix,body}); return suffix.endsWith("/cancel") ? finished("cancelled") : running();}});
  await h.actions.start([0]); await h.actions.cancel();
  assert.deepEqual(calls[1],{id:"project-a",suffix:"/music-emotion/cancel",body:{}});
  assert.equal(h.timers.size,0); assert.equal(h.refreshed,1); assert.equal(h.actions.busy,false); assert.equal(h.actions.cancelling,false);
});

test("GET failure stops automatic retry and manual refresh recovers", async () => {
  let fail = true, requests = 0;
  const h = harness({requestProject:async () => {requests += 1; if (fail) throw new Error("服务不可用"); return running();}});
  await h.actions.connect(h.state.plan);
  assert.match(h.actions.error,/刷新情绪状态/); assert.equal(h.timers.size,0); assert.equal(requests,1);
  fail = false; await h.actions.refreshStatus();
  assert.equal(h.actions.error,""); assert.equal(h.timers.size,1); assert.equal(requests,2);
});

test("a failed director brief refresh can be retried after job completion", async () => {
  let fail = true, refreshes = 0;
  const h = harness({requestProject:async () => finished("completed"), refresh:async () => {refreshes += 1; if (fail) throw new Error("项目读取失败");}});
  await h.actions.connect(h.state.plan);
  assert.match(h.actions.error,/项目读取失败/); assert.equal(refreshes,1);
  fail = false; await h.actions.refreshStatus(); assert.equal(refreshes,2); assert.equal(h.actions.error,"");
});

test("cancel failure keeps the active job available for an explicit retry", async () => {
  let fail = true;
  const h = harness({requestProject:async (id,suffix) => {if (suffix.endsWith("/cancel")) {if (fail) throw new Error("连接失败"); return finished("cancelled");} return running();}});
  await h.actions.connect(h.state.plan);
  await assert.rejects(h.actions.cancel(),/连接失败/);
  assert.equal(h.actions.cancelling,false); assert.equal(h.actions.busy,true); assert.equal(h.timers.size,0);
  fail = false; await h.actions.cancel(); assert.equal(h.actions.busy,false);
});

test("late status reads cannot replace a newly submitted job", async () => {
  const old = deferred();
  const h = harness({requestProject:async (id,suffix,body) => body ? running() : old.promise});
  const connecting = h.actions.connect(h.state.plan);
  await h.actions.start([0]); old.resolve({ready:true,job:null}); await connecting;
  assert.equal(h.actions.result.job.status,"running"); assert.equal(h.timers.size,1);
});

test("generation and expansion block analysis, and intensity is explicitly an estimate", () => {
  assert.match(musicEmotionBlocked({projectId:"project-a",plan:{...project(),expansion_active:true}}),/批量扩写/);
  assert.match(musicEmotionBlocked({projectId:"project-a",plan:{...project(),controller_active:true}}),/视频生成/);
  assert.match(musicEmotionSummary({music_emotion:{status:"ready",analysis:{labels:["克制","期待"],intensity:3,summary:"逐渐推高"}}}),/强度 3\/5（模型估计）/);
  assert.match(musicEmotionSummary({music_emotion:{status:"stale"}}),/重新分析/);
});
