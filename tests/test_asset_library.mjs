import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const plain = value => JSON.parse(JSON.stringify(value));
function harness() {
  const elements = [];
  const document = {body:{append() {}}, createElement(tag) {
    const item = {tag, value:"", children:[], isConnected:true, classList:{toggle() {}},
      append(child) {this.children.push(child);}, setAttribute(name, value) {this[name] = value;},
      remove() {this.isConnected = false;}, replaceChildren() {this.children = [];}};
    elements.push(item); return item;
  }};
  const context = {document, URLSearchParams, FormData, window:{confirm:() => true, alert() {}},
    withUiLogging: action => action, api:{apiURL:path => path}};
  const mentions = fs.readFileSync(new URL("../web/material_mentions.js", import.meta.url), "utf8").replaceAll("export ", "");
  const source = fs.readFileSync(new URL("../web/asset_library.js", import.meta.url), "utf8").replace(/^import .*;\r?\n/gm, "").replaceAll("export ", "");
  vm.createContext(context); vm.runInContext(mentions+"\n"+source+"\nglobalThis.lib = {toggleLibrarySelection,mergeLibraryImport,openAssetLibrary};", context);
  return {context, elements};
}

test("library selection preserves click order across unselect and reselect", () => {
  const {context} = harness(), selected = [];
  context.lib.toggleLibrarySelection(selected, {id:"b", version:2});
  context.lib.toggleLibrarySelection(selected, {id:"a", version:1});
  context.lib.toggleLibrarySelection(selected, {id:"b", version:3});
  context.lib.toggleLibrarySelection(selected, {id:"b", version:3});
  assert.deepEqual(plain(selected), [{asset_id:"a",version:1},{asset_id:"b",version:3}]);
});

test("import appends descriptions at final image numbers and protects old references", () => {
  const {context} = harness(), merge = context.lib.mergeLibraryImport;
  const imported = {refs:["scene"], origins:{scene:{name_at_import:"舞台",description:"灯光 @图1"}}};
  assert.deepEqual(plain(merge(["person"], "@图1 是主角", imported)), {refs:["person","scene"], note:"@图1 是主角\n@图2：舞台；灯光 参考图"});
  assert.throws(() => merge(["person"], "@图1 是主角", imported, {mode:"replace", noteMode:"keep"}), /仍引用/);
  assert.deepEqual(plain(merge(["person"], "@图1 是主角", imported, {mode:"replace",noteMode:"replace"})), {refs:["scene"],note:"@图1：舞台；灯光 参考图"});
  assert.throws(() => merge(["1","2","3","4","5","6"], "", imported), /最多 6/);
});

test("replacement remaps retained references by image identity", () => {
  const {context} = harness();
  const result = context.lib.mergeLibraryImport(["a","b"], "@图2 是场景", {refs:["b","c"]}, {mode:"replace", noteMode:"keep"});
  assert.deepEqual(plain(result), {refs:["b","c"], note:"@图1 是场景"});
});

test("library modal imports selected snapshots once in click order with project revision", async () => {
  const {context,elements} = harness(), calls = [], imported = [];
  context.api.fetchApi = async (path, init) => {
    calls.push({path, body:init?.body && JSON.parse(init.body)});
    return {ok:true,json:async () => path.includes("/library/import") ? {refs:["b.png","a.png"]} : {revision:8, assets:[
      {id:"a",version:1,name:"角色",category:"character",tags:[]},
      {id:"b",version:2,name:"舞台",category:"scene",tags:[]},
    ]}};
  };
  await context.lib.openAssetLibrary({projectId:"p",getRevision:() => 4,onImport:(data,choices) => imported.push({data,choices})});
  let picks = elements.filter(item => item.textContent === "选择");
  picks[1].onclick(); await new Promise(resolve => setImmediate(resolve));
  picks = elements.filter(item => item.textContent === "选择");
  picks.at(-1).onclick(); await new Promise(resolve => setImmediate(resolve));
  elements.find(item => item.textContent === "参考图").children[0].value = "append";
  elements.find(item => item.textContent === "素材说明").children[0].value = "append";
  await elements.find(item => item.textContent === "导入所选素材").onclick();
  const writes = calls.filter(item => item.path.includes("/library/import"));
  assert.equal(writes.length, 1);
  assert.deepEqual(plain(writes[0].body), {revision:4,selection:[{asset_id:"b",version:2},{asset_id:"a",version:1}]});
  assert.equal(imported.length, 1);
  assert.deepEqual(plain(imported[0].choices), {mode:"append",noteMode:"append"});
});

test("inherited material editor stays read only and only custom imports mark draft changed", async () => {
  const h = harness(); let changed = 0, opened;
  h.context.api.apiURL = path => path;
  const source = fs.readFileSync(new URL("../web/materials.js",import.meta.url),"utf8").replace(/^import .*;\r?\n/gm,"").replaceAll("export ","");
  vm.runInContext(source+"\nglobalThis.makeEditor = materialEditor;",h.context);
  vm.runInContext("openAssetLibrary = async value => {globalThis.opened = value;};",h.context);
  const parent = h.context.document.createElement("div");
  const editor = h.context.makeEditor(parent,{projectId:"p",refs:[],note:"",defaults:() => ({refs:["default.png"],note:"默认说明"}),source:"default",getRevision:() => 9,changed:() => changed++});
  const library = h.elements.find(item => item.textContent === "从素材库选择");
  assert.equal(library.disabled,true); assert.equal(editor.note.disabled,true);
  editor.source.value = "custom"; editor.source.onchange(); assert.equal(library.disabled,false);
  await library.onclick(); opened = h.context.opened;
  assert.equal(opened.getRevision(),9); assert.equal(opened.projectId,"p");
  await opened.onImport({refs:["scene.png"],origins:{"scene.png":{name_at_import:"舞台"}}},{mode:"append",noteMode:"append"});
  assert.deepEqual(plain(editor.refs),["scene.png"]); assert.equal(editor.getNote(),"@图1：舞台");
  assert.equal(changed,2);
});
