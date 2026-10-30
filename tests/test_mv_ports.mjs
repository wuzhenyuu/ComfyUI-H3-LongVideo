import {alignProjectNode} from '../web/node_ui.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const {syncHiddenWidgetConnections}=await import('data:text/javascript,'+encodeURIComponent(fs.readFileSync(new URL('../web/widget_visibility.js',import.meta.url),'utf8')));
const videoMaterials=fs.readFileSync(new URL('../web/video_materials.js',import.meta.url),'utf8').replace(/^import .*;\r?\n/gm,'').replace(/^export /gm,'');
const appendVideoOutputs=vm.runInNewContext(videoMaterials+'\nappendVideoOutputs');

async function registeredNode(inputSchema) {
 let extension;
 const script=fs.readFileSync(new URL('../web/h3lv.js',import.meta.url),'utf8').replace(/^import .*;\r?\n/gm,'').replace('new URL("./h3lv.css", import.meta.url).href','"h3lv.css"');
 vm.runInNewContext(script,{alignProjectNode,app:{registerExtension:e=>extension=e},document:{querySelector:()=>({})},syncHiddenWidgetConnections,appendVideoOutputs,setTimeout:()=>{},setWorkflowProgress:()=>{}});
 function Node(){this.inputs=[{name:'audio',link:10},{name:'vocals',link:11},{name:'mode',widget:{},link:null}];this.widgets=[];this.removed=[];
  this.outputs=Array.from({length:13},(_,index)=>({name:`legacy_${index}`,links:[index+20]}));}
 Node.prototype.addOutput=function(name,type){this.outputs.push({name,type});};
 Node.prototype.addWidget=function(type,name,value,callback){const widget={type,name,value,callback};this.widgets.push(widget);return widget;};
 Node.prototype.removeInput=function(index){this.removed.push(this.inputs[index]);this.inputs.splice(index,1);};
 Node.prototype.onConfigure=function(){return 'configured';};
 await extension.beforeRegisterNodeDef(Node,{name:'H3LVUnified',...(inputSchema===undefined?{}:{input:inputSchema})});
 return Node;
}

test('loading an old MV graph removes vocals only when the server explicitly lacks that port',async()=>{
 const Node=await registeredNode({required:{audio:['AUDIO']},optional:{}});
 const node=new Node();assert.equal(node.onConfigure({widgets_values:[]}), 'configured');
 assert.deepEqual(node.inputs.map(i=>i.name),['audio','mode']);
 assert.equal(node.inputs[0].link,10);assert.equal(node.removed[0].link,11);
 assert.deepEqual(node.outputs.slice(0,13).map(item=>item.links[0]),Array.from({length:13},(_,index)=>index+20));
 assert.deepEqual(node.outputs.slice(13).map(item=>item.name),['video_1','video_2','video_3']);
 node.onConfigure({widgets_values:[]});assert.equal(node.removed.length,1);
 assert.equal(node.outputs.length,16);
});

test('new optional-vocals schema preserves old external connections and every output slot',async()=>{
 const Node=await registeredNode({required:{audio:['AUDIO']},optional:{vocals:['AUDIO',{}]}});
 const node=new Node();node.onNodeCreated();
 assert.deepEqual(node.inputs.map(input=>[input.name,input.link]),[['audio',10],['vocals',11],['mode',null]]);
 assert.equal(node.onConfigure({widgets_values:[]}), 'configured');
 node.onConfigure({widgets_values:[]});
 assert.deepEqual(node.inputs.map(input=>[input.name,input.link]),[['audio',10],['vocals',11],['mode',null]]);
 assert.equal(node.removed.length,0);
 assert.deepEqual(node.outputs.slice(0,13).map(item=>item.links[0]),Array.from({length:13},(_,index)=>index+20));
 assert.equal(node.outputs.length,16);
});

test('missing server schema never guesses that a saved vocal connection is obsolete',async()=>{
 const Node=await registeredNode();const node=new Node();
 node.onNodeCreated();node.onConfigure({widgets_values:[]});
 assert.equal(node.inputs.find(input=>input.name==='vocals').link,11);
 assert.equal(node.removed.length,0);
});

test('explicit no-vocals schema also migrates the creation path through native slot removal',async()=>{
 const Node=await registeredNode({required:{audio:['AUDIO']},optional:{}});const node=new Node();
 node.onNodeCreated();node.onConfigure({widgets_values:[]});
 assert.deepEqual(node.inputs.map(input=>input.name),['audio','mode']);
 assert.equal(node.removed.length,1);assert.equal(node.removed[0].link,11);
});
