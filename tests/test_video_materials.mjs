import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
const plain = value => JSON.parse(JSON.stringify(value));
const flush = () => new Promise(resolve => setImmediate(resolve));
function harness() {
  const elements = [], calls = [], alerts = [];
  const document = {body:{append() {}},createElement(tag) {
    const node = {tag,value:"",children:[],isConnected:true,classList:{toggle() {}},selectionStart:0,selectionEnd:0,
      append(child) {this.children.push(child);child.parent=this;},setAttribute(key,value) {this[key]=value;},
      remove() {this.isConnected=false;},replaceChildren() {this.children=[];},focus() {},setSelectionRange(start,end) {this.selectionStart=start;this.selectionEnd=end;}};
    elements.push(node); return node;
  }};
  const context = {document,URLSearchParams,FormData,window:{confirm:()=>true,alert:value=>alerts.push(value)},withUiLogging:fn=>fn,
    api:{apiURL:path=>path,fetchApi:async (path,init) => {calls.push({path,body:init?.body});return {ok:true,json:async()=>({duration:3})};}}};
  const read = name => fs.readFileSync(new URL(`../web/${name}.js`,import.meta.url),"utf8").replace(/^import .*;\r?\n/gm,"").replaceAll("export ","");
  vm.createContext(context);
  vm.runInContext(read("material_mentions")+"\n"+read("asset_library")+"\n"+read("video_materials")+"\nglobalThis.testing={videoMaterialEditor,hasVideoConfiguration,appendVideoOutputs,validateVideoReferences,mergeVideoLibraryImport,remapVideoMentions,openAssetLibrary};",context);
  return {context,elements,calls,alerts,make:options=>context.testing.videoMaterialEditor(document.createElement("div"),{projectId:"p",getRevision:()=>8,...options}),find:text=>elements.find(item=>item.textContent===text)};
}

test("opening an old empty project and rendering does not opt into video management", async () => {
  const h = harness(), t=h.context.testing;
  assert.equal(t.hasVideoConfiguration({}),false); assert.equal(t.hasVideoConfiguration({},{}),false);
  const defaults=h.make({}), row=h.make({defaults:()=>defaults.getState()}); await defaults.ready;await row.ready;
  defaults.render();row.render(); assert.deepEqual(plain(defaults.getDefaultPayload()),{});assert.deepEqual(plain(row.getPayload()),{});
  assert.equal(h.calls.length,0);
  assert.equal(t.hasVideoConfiguration({video_materials_version:1}),true);
  assert.equal(t.hasVideoConfiguration({default_video_refs:[]}),true);
});

test("video inheritance and custom override are independent and explicit empty remains saved", async () => {
  const h=harness();let changes=0;
  const defaults=h.make({configured:true,refs:["a.mp4"],metadata:{"a.mp4":{duration:3}},note:"@视频1 推镜"});
  const row=h.make({defaults:()=>defaults.getState(),changed:()=>changes++});await row.ready;
  assert.equal(row.note.disabled,true);assert.equal(row.note.value,"@视频1 推镜");
  row.source.value="custom";row.source.onchange();assert.equal(row.note.disabled,false);
  assert.deepEqual(plain(row.getPayload()),{video_refs:[],video_note:"",video_reference_source:"custom"});
  row.source.value="default";row.source.onchange();assert.equal(row.note.value,"@视频1 推镜");
  assert.equal(row.getPayload().video_reference_source,"default");assert.equal(changes,2);
});

test("reorder remaps mentions by file identity and referenced videos cannot be removed", async () => {
  const h=harness(), editor=h.make({configured:true,refs:["a","b"],metadata:{a:{duration:3},b:{duration:4}},note:"@视频1 推镜，<Video 2> 动作"});await editor.ready;
  h.elements.filter(item=>item.textContent==="下移").at(-2).onclick();
  assert.deepEqual(plain(editor.refs),["b","a"]);assert.equal(editor.getPayload().video_note,"@视频2 推镜，<Video 1> 动作");
  h.elements.filter(item=>item.textContent==="移除").at(-2).onclick();assert.equal(editor.refs.length,2);assert.match(h.alerts.at(-1),/正在引用/);
  editor.note.value="";editor.note.oninput();h.elements.filter(item=>item.textContent==="移除").at(-2).onclick();
  assert.deepEqual(plain(editor.refs),["a"]);
});

test("video limits validate finite individual lengths, count and total duration", () => {
  const validate=harness().context.testing.validateVideoReferences;
  assert.equal(validate(["a","b","c"],{a:{duration:2},b:{duration:3},c:{duration:10}}),15);
  for (const duration of [0,1.99,15.01,NaN,Infinity]) assert.throws(()=>validate(["a"],{a:{duration}}),/2–15/);
  assert.throws(()=>validate(["a","b"],{a:{duration:8},b:{duration:8}}),/总时长/);
  assert.throws(()=>validate(["a","b","c","d"],{}),/最多 3/);
});

test("library video append and replace validate final set and remap retained mentions", () => {
  const merge=harness().context.testing.mergeVideoLibraryImport;
  const imported={video_refs:["b"],video_metadata:{b:{duration:8}},origins:{b:{name_at_import:"动作",description:"跟随 @视频1"}}};
  const result=merge(["a"],"@视频1 推镜",{a:{duration:3}},imported);
  assert.deepEqual(plain(result.refs),["a","b"]);assert.equal(result.note,"@视频1 推镜\n@视频2：动作；跟随 参考视频");
  assert.throws(()=>merge(["a"],"",{a:{duration:8}},imported),/总时长/);
  assert.throws(()=>merge(["a"],"@视频1 推镜",{a:{duration:3}},imported,{mode:"replace",noteMode:"keep"}),/仍引用/);
  assert.equal(merge(["a","b"],"@视频2 动作",{a:{duration:3},b:{duration:8}},imported,{mode:"replace",noteMode:"keep"}).note,"@视频1 动作");
  assert.deepEqual(plain(merge(["a"],"@视频1",{a:{duration:3}},imported,{mode:"replace",noteMode:"replace"}).refs),["b"]);
});

test("readonly video editor locks selection, file input, notes and mutation handlers", async () => {
  const h=harness();let changes=0;
  const e=h.make({configured:true,readOnly:true,refs:["a"],metadata:{a:{duration:3}},changed:()=>changes++});await e.ready;
  for(const node of h.elements.filter(item=>["button","input","textarea","select"].includes(item.tag)))assert.equal(node.disabled,true);
  await h.find("从视频素材库选择").onclick();h.find("移除").onclick();e.note.value="changed";e.note.oninput();
  assert.equal(changes,0);assert.equal(h.calls.length,0);assert.equal(e.getPayload().video_note,"");
  e.setReadOnly(false);assert.equal(e.note.disabled,false);
});

test("existing refs metadata hydrates without marking draft changed", async () => {
  const h=harness();let changes=0;const e=h.make({refs:["clip.mp4"],configured:true,changed:()=>changes++});await e.ready;
  assert.equal(e.getState().metadata["clip.mp4"].duration,3);assert.equal(changes,0);
  assert.equal(h.calls[0].path,"/h3lv/project/p/video-refs/clip.mp4/info");
  const preview=h.elements.find(item=>item.tag==="video");assert.equal(preview.muted,true);assert.equal(preview.preload,"metadata");
});

test("video library filters media in library and trash and imports typed immutable selections", async () => {
  const h=harness(), writes=[];h.context.api.fetchApi=async(path,init)=>{
    h.calls.push({path,body:init?.body});
    if(path.includes("/library/import")){writes.push(JSON.parse(init.body));return{ok:true,json:async()=>({video_refs:["a"],video_metadata:{a:{duration:3}}})};}
    return{ok:true,json:async()=>({revision:7,assets:[{id:"i",name:"image",version:1,category:"scene"},{id:"a",name:"video",version:2,media_type:"video",category:"video",duration:3,tags:[]}]})};
  };
  let imported;
  await h.context.testing.openAssetLibrary({projectId:"p",mediaType:"video",getRevision:()=>9,onImport:result=>imported=result});
  assert.match(h.calls[0].path,/media_type=video/);assert.equal(h.elements.some(item=>item.tag==="strong"&&item.textContent==="image"),false);
  h.find("选择").onclick();await flush();
  h.find("显示").children[0].value="1";h.find("显示").children[0].onchange();await flush();
  assert.ok(h.calls.some(item=>item.path.includes("deleted=1")&&item.path.includes("media_type=video")));
  await h.find("导入所选素材").onclick();assert.deepEqual(writes,[{revision:9,selection:[{asset_id:"a",version:2}],media_type:"video"}]);
  assert.deepEqual(plain(imported.video_refs),["a"]);
});

test("old graph output migration appends three IMAGE slots without disturbing original links", () => {
  const migrate=harness().context.testing.appendVideoOutputs;
  const outputs=Array.from({length:13},(_,i)=>({name:`original_${i}`,type:"old",links:[i+100]}));const original=[...outputs];
  const node={outputs,addOutput(name,type){this.outputs.push({name,type});}};
  migrate(node);assert.equal(outputs.length,16);original.forEach((item,i)=>assert.equal(outputs[i],item));
  assert.deepEqual(outputs.slice(13),[{name:"video_1",type:"IMAGE"},{name:"video_2",type:"IMAGE"},{name:"video_3",type:"IMAGE"}]);
  migrate(node);assert.equal(outputs.length,16);
});

test("video upload validates the final batch before committing references or opting in", async () => {
  const h=harness();let changes=0,sequence=0;const e=h.make({changed:()=>changes++});await e.ready;
  h.context.api.fetchApi=async()=>({ok:true,json:async()=>({name:`clip${++sequence}.mp4`,metadata:{duration:8}})});
  const upload=h.elements.find(item=>item.tag==="input"&&item.type==="file");
  upload.files=[new File(["a"],"a.mp4"),new File(["b"],"b.mp4")];await upload.onchange();
  assert.match(h.alerts.at(-1),/总时长/);assert.deepEqual(plain(e.getPayload()),{});assert.equal(changes,0);
  upload.files=[new File(["c"],"c.mp4")];await upload.onchange();
  assert.deepEqual(plain(e.getPayload()),{video_refs:["clip3.mp4"],video_note:""});assert.equal(changes,1);
});

test("saving catches dangling video mentions before submitting the draft", async () => {
  const h=harness(),e=h.make({configured:true,refs:["a"],metadata:{a:{duration:3}}});await e.ready;
  e.note.value="@视频2";e.note.oninput();assert.throws(()=>e.getDefaultPayload(),/没有对应视频/);
});

test("legacy material plans keep both video editors readonly across metadata render", async () => {
  const source=fs.readFileSync(new URL("../web/h3lv.js",import.meta.url),"utf8");
  const readOnlyExpressions=[...source.matchAll(/readOnly:([^,]+), changed:/g)].map(match=>match[1]);
  assert.equal(readOnlyExpressions.length,2);
  for(const expression of readOnlyExpressions){
    const locked=plan=>vm.runInNewContext(expression,{plan});
    for(const plan of [{},{materials_version:0},{expansion_active:true},{materials_version:1,expansion_active:true}])assert.equal(locked(plan),true);
    assert.equal(locked({materials_version:1}),false);
    const h=harness();let changes=0;const editor=h.make({readOnly:locked({}),defaults:()=>({refs:[],note:"",metadata:{}}),changed:()=>changes++});await editor.ready;
    editor.render();assert.equal(editor.source.disabled,true);assert.equal(editor.note.disabled,true);
    editor.source.value="custom";editor.source.onchange();await h.find("从视频素材库选择").onclick();
    assert.equal(changes,0);assert.deepEqual(plain(editor.getPayload()),{});assert.equal(h.calls.length,0);
  }
  assert.match(source,/参考视频暂只读：请先.*启用内置素材管理/);
  assert.doesNotMatch(source,/\.setReadOnly\(false\)/);
});
