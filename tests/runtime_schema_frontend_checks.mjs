import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = name => fs.readFileSync(new URL(`../web/js/${name}.js`, import.meta.url), 'utf8')
  .replace(/^import .*;\r?$/gm, '').replace(/^export /gm, '');
const model = {id:'m', backend:'llamacpp', status:'offline', in_ini:true, settings:{}};
let engine = 'llamacpp', target = 'llamacpp', binary = '/builtin/server';
let customName = 'strix-llama.cpp_vulkan';
const state = () => ({active_engine:engine,
  active_build:{id:engine==='ikllama'?'ikllama':target,
    name:engine==='ikllama'?'ik_llama':target.startsWith('custom')?customName:target?'llama.cpp':'External / unregistered build',
    server_bin:engine==='ikllama'?'/ik/server':binary},
  config:{active_engine:engine,
  active_llamacpp_build_target:target, server_bin:binary, ik_llama_server_bin:'/ik/server'},
  models:[{...model}], gpus:[], presets:{}, preset_bindings:{}});
const schema = () => ({count:1, groups:[{name:'Options', knobs:[{key:
  engine === 'ikllama' ? 'ik-only' : target.startsWith('custom') ? 'fork-only' : 'builtin-only',
  type:'str', aliases:[], desc:'test option'}]}]});
let requests = [], failSwitch = false, deferState = null, deferSchema = null;
async function api(path, body) {
  requests.push(path);
  if(path === '/api/schema') {
    if(deferSchema) { const deferred=deferSchema;deferSchema=null;return await deferred; }
    return schema();
  }
  if(path === '/api/state') {
    if(deferState) { const deferred = deferState; deferState = null; return await deferred; }
    return state();
  }
  if(path === '/api/build/activate') {
    if(failSwitch) return {ok:false,error:'refused'};
    engine='llamacpp';target=body.target;binary='/custom/server';return {ok:true};
  }
  if(path === '/api/engine/switch') {
    if(failSwitch) return {ok:false,error:'refused'};
    engine=body.engine;target='llamacpp';binary='/builtin/server';return {ok:true};
  }
  if(path === '/api/build/targets') return {targets:[
    {id:'llamacpp',name:'llama.cpp',builtin:true}, {id:'ikllama',name:'ik_llama',builtin:true},
    {id:'custom-one',name:'Fork',builtin:false}], active_build:{id:engine==='ikllama'?'ikllama':target}};
  if(path.startsWith('/api/build/info') || path.startsWith('/api/build/log')) return {};
  if(path.startsWith('/api/vllm/version')) return {error:'unsupported'};
  throw Error(path);
}
const edit = {innerHTML:'', input:null};
let builds = 0;
const S = {STATE:state(), SCHEMA:schema(), VLLM_SCHEMA:null};
const listeners={};
const on=(event, fn)=>(listeners[event] ||= []).push(fn);
const emit=(event,...args)=>{for(const fn of listeners[event] || []) fn(...args);};
const badge={innerHTML:''};
let badgeWrites=0;
const badgeCtx=vm.createContext({S,on,$:()=>badge,
  setHTML:(node,html)=>{node.innerHTML=html;badgeWrites++;}});
const core=source('core');
vm.runInContext(core.slice(core.indexOf('const esc ='),core.indexOf('// Every value interpolated')),badgeCtx);
const main=source('main');
vm.runInContext(main.slice(main.indexOf('const ENGINE_LABEL'),main.indexOf('function clock()')),badgeCtx);
badgeCtx.renderEngineBadge();
assert.match(badge.innerHTML,/>llama.cpp<\/span>/);
const ctx = vm.createContext({S, api, console, row:{lastElementChild:edit},
  modelRows:()=>S.STATE.models, cfgOf:()=>S.STATE.config,
  $:()=>null, $$:()=>[], esc:String,
  setHTML:(node, html)=>{node.innerHTML=html;if(node===edit){builds++;node.input={value:'server default'};}},
  on, emit, activeTab:()=> 'build', initAutoTune:()=>{}, syncAutoTune:()=>{},
  toast:()=>{}, meter:()=>'', localStorage:{getItem:()=>null,setItem:()=>{},removeItem:()=>{}},
  document:{title:''}, setTimeout:()=>{}, clearTimeout:()=>{}, CSS:{escape:String},
});
vm.runInContext(source('models') + `
setOpenId('m');
renderGpus = () => {};
updateCmpRun = () => {};
renderModels = () => syncEditor(row, S.STATE.models[0]);
renderModels();
`, ctx);
assert.match(edit.innerHTML, /data-k="builtin-only"/);
const firstInput = edit.input;
firstInput.value='half typed';
const beforePoll = builds;
await ctx.refresh(true);
assert.equal(edit.input, firstInput);
assert.equal(edit.input.value, 'half typed');
assert.equal(builds, beforePoll, 'ordinary 4-second refresh preserves input DOM');
assert.equal(badgeWrites,1,'unchanged state must not churn badge DOM');

const nodes = {};
const build = vm.createContext({api, refreshRuntime:ctx.refreshRuntime,
  $:s=>nodes[s] ||= {style:{}}, esc:String, setHTML:(n,h)=>n.innerHTML=h,
  localStorage:{getItem:()=> 'custom-one',setItem:()=>{}},
  setInterval:()=>1, clearInterval:()=>{}, setTimeout:()=>{}, toast:()=>{},
  agoText:()=>'', fmtDur:()=>'', confirm:()=>true,
});
vm.runInContext(source('build'), build);
await build.loadBuild();
requests=[];
await nodes['#btn-use-build'].onclick();
assert.ok(requests.includes('/api/schema'));
assert.ok(requests.includes('/api/state'));
assert.match(edit.innerHTML, /data-k="fork-only"/);
assert.doesNotMatch(edit.innerHTML, /data-k="builtin-only"/);
assert.notEqual(edit.input, firstInput, 'same-engine custom switch discards old editor');

// Returning to Models needs no loader or full-page reload: hidden rows are ready.
assert.equal(S.STATE.config.active_llamacpp_build_target, 'custom-one');
assert.match(badge.innerHTML,/>strix-llama.cpp_vulkan<\/span>/,'badge updates within successful switch');
await vm.runInContext('setTarget("ikllama"); loadBuild()', build);
requests=[];
await nodes['#btn-switch-engine'].onclick();
assert.ok(requests.includes('/api/schema'));
assert.match(edit.innerHTML, /data-k="ik-only"/);
assert.match(badge.innerHTML,/>ik_llama<\/span>/);
assert.doesNotMatch(edit.innerHTML, /data-k="fork-only"/);
await vm.runInContext('setTarget("llamacpp"); loadBuild()', build);
await nodes['#btn-switch-engine'].onclick();
assert.match(edit.innerHTML, /data-k="builtin-only"/);
assert.doesNotMatch(edit.innerHTML, /data-k="ik-only"/);
assert.match(badge.innerHTML,/>llama.cpp<\/span>/);

// Failed switches do not invalidate a working editor or fetch a new schema.
await vm.runInContext('setTarget("custom-one"); loadBuild()', build);
failSwitch=true;requests=[];
const currentInput=edit.input;
await nodes['#btn-use-build'].onclick();
assert.equal(edit.input,currentInput);
assert.ok(!requests.includes('/api/schema'));
failSwitch=false;

// Identity changes discovered by polling include target ID and binary path.
let previousInput=edit.input;
target='custom-two';
await ctx.refresh(true);
assert.notEqual(edit.input,previousInput);
assert.match(edit.innerHTML,/data-k="fork-only"/);
previousInput=edit.input;binary='/different/server';requests=[];
await ctx.refresh(true);
assert.notEqual(edit.input,previousInput);
assert.ok(requests.includes('/api/schema'));

// An in-flight old poll cannot overwrite an explicit boundary refresh.
let resolveOld;
deferState=new Promise(resolve=>resolveOld=resolve);
const oldState=state();
const oldPoll=ctx.refresh(true);
engine='ikllama';
await ctx.refreshRuntime();
resolveOld(oldState);
await oldPoll;
assert.equal(S.STATE.active_engine,'ikllama');
assert.match(edit.innerHTML,/data-k="ik-only"/);

// A delayed old schema response also cannot repopulate the cleared cache.
let resolveSchema;
const oldSchema=schema();
S.SCHEMA=null;
deferSchema=new Promise(resolve=>resolveSchema=resolve);
const schemaPoll=ctx.refresh(true);
for(let i=0;i<10 && deferSchema;i++) await Promise.resolve();
assert.equal(deferSchema,null,'old poll has reached schema discovery');
engine='llamacpp';target='llamacpp';binary='/builtin/server';
await ctx.refreshRuntime();
resolveSchema(oldSchema);
await schemaPoll;
assert.match(edit.innerHTML,/data-k="builtin-only"/);
assert.doesNotMatch(edit.innerHTML,/data-k="ik-only"/);

// Slow normal polls can complete while a newer poll is still pending.
let resolveFirst, resolveSecond;
deferState=new Promise(resolve=>resolveFirst=resolve);
const firstPoll=ctx.refresh(true);
deferState=new Promise(resolve=>resolveSecond=resolve);
const secondPoll=ctx.refresh(true);
const stateBefore=S.STATE;
resolveFirst(state());
await firstPoll;
assert.notEqual(S.STATE,stateBefore,'overlapping polls must not starve updates');
resolveSecond(state());
await secondPoll;

// Backend-resolved external identity and renamed targets update without reload.
target='';binary='/external/server';
await ctx.refresh(true);
assert.match(badge.innerHTML,/>External \/ unregistered build<\/span>/);
target='custom-one';customName='Renamed <fork> & "build"';
await ctx.refresh(true);
assert.match(badge.innerHTML,/Renamed &lt;fork&gt; &amp; &quot;build&quot;/);
assert.ok(!badge.innerHTML.includes('<fork>'));
customName='Second name';
await ctx.refresh(true);
assert.match(badge.innerHTML,/>Second name<\/span>/);

// vLLM schema cache is independent from llama-family switches.
const vllm={count:1, groups:[]};S.VLLM_SCHEMA=vllm;
await ctx.refreshRuntime();
assert.equal(S.VLLM_SCHEMA,vllm);
console.log('runtime schema refresh checks passed');
