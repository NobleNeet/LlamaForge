import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source = fs.readFileSync(new URL('../web/js/stats.js', import.meta.url), 'utf8')
  .replace(/^import .*;$/m, '').replaceAll('export ', '');
const handlers = {}, calls = [];
const view = {innerHTML:'',addEventListener:(kind, fn)=>handlers[kind]=fn};
const esc = v => String(v ?? '').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cfg = {id:3, loaded_at:1790980140, engine_name:'strix-llama', engine_commit:'06a64c312345',engine_executable:'/source/llama-server',
 normalized_options:{performance:{'Context (tok)':'65536'},speculative:{Type:'draft-mtp',Adaptive:'on'},models:{Main:'/models/a b.gguf',Draft:'mtp.gguf',MMProj:'mmproj.gguf'}},
 other_options:['--fork-option','<script>'], argv:['/source/llama-server','--model',"/models/a b'c.gguf",'--api-key','[REDACTED]']};
const history = {configs:{3:cfg},runs:[{timestamp:1790980141,load_config_id:3,prompt_tokens:22627,pp_tps:244.34,generated_tokens:1297,tg_tps:14.03,total_ms:184990,mtp_acceptance:.51855,mtp_accepted:699,mtp_generated:1348,mtp_mean_len:2.17},
 {timestamp:1790980041,load_config_id:3,prompt_tokens:null,pp_tps:null,generated_tokens:0,tg_tps:null,total_ms:null,mtp_acceptance:null}]};
const summary = {per_model:[{id:'a & b',has_history:true,tokens:1,prompt:1,generated:0,runs:1,avg_tps:0,loaded_secs:0,last_used:0}],totals:{},live:{loaded_models:[]},daily:[]};
const context = vm.createContext({Date, console, encodeURIComponent, $, esc, setHTML:(el, html)=>el.innerHTML=html,
 api:async (path, body)=>{calls.push({path,body}); if(path==='/api/stats') return summary; if(path==='/api/stats/reset')return {ok:true};return history;},
 fmtNum:String, fmtDur:String,fmtAgo:String,toast:()=>{},confirm:()=>true});
function $(selector){assert.equal(selector,'#view-stats');return view;}
vm.runInContext(source,context);
await vm.runInContext('initStats(); loadStats()',context);
assert.doesNotMatch(view.innerHTML,/RECENT RUNS/);
assert.match(view.innerHTML,/data-historymodel="a &amp; b"/);
function click(selector, dataset={}) {
 const target = {closest:q=>q===selector ? {dataset}:null};
 handlers.click({target});
}
async function settled(){ for(let i=0;i<10;i++) await Promise.resolve(); }
click('[data-historymodel]',{historymodel:'a & b'});
await settled();
assert.match(calls.at(-1).path,/model=a%20%26%20b&limit=10$/);
assert.match(view.innerHTML,/RECENT RUNS/);
assert.match(view.innerHTML,/Latest 10/);
for(const n of [10,25,50,100])assert.match(view.innerHTML,new RegExp(`value="${n}"`));
assert.match(view.innerHTML,/<td title="\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}">\d{2}:\d{2}<\/td>/);
assert.match(view.innerHTML,/Acceptance: 51\.855%.*699 accepted \/ 1348 generated.*mean length: 2\.17/);
assert.match(view.innerHTML,/>51\.9<\/td>/);
assert.match(view.innerHTML,/>—<\/td>/);
const columns = ['Time','Prompt (tok)','PP (tok/s)','Generated (tok)','TG (tok/s)','Total (s)','MTP Accept (%)','Load Config'];
let last = -1;
for(const col of columns){const pos=view.innerHTML.indexOf(`<th>${col}</th>`);assert.ok(pos>last);last=pos;}
assert.match(view.innerHTML,/strix-llama \(06a64c3\)/);
assert.match(view.innerHTML,/ENGINE-SPECIFIC \/ OTHER OPTIONS/);
assert.match(view.innerHTML,/&lt;script&gt;/);
assert.match(view.innerHTML,/Main/);
assert.doesNotMatch(view.innerHTML,/FULL LAUNCH COMMAND/);
click('[data-loadconfig]',{loadconfig:'3'}); await settled();
click('[data-command]'); await settled();
assert.match(view.innerHTML,/FULL LAUNCH COMMAND/);
assert.match(view.innerHTML,/\[REDACTED\]/);
const command = vm.runInContext(`quoteArg("a'b")`, context);
assert.equal(command, `'a'"'"'b'`, 'shell quoting must preserve a literal quote');
for (const n of [25,50,100,10]) {
 handlers.change({target:{matches:q=>q==='[data-runlimit]',value:String(n)}});await settled();
 assert.match(calls.at(-1).path,new RegExp(`limit=${n}$`));
}
await vm.runInContext('loadStats(true)',context);
assert.match(view.innerHTML,/FULL LAUNCH COMMAND/, 'refresh preserves full command expansion');
click('[data-historymodel]',{historymodel:'a & b'});await settled();
assert.doesNotMatch(view.innerHTML,/RECENT RUNS/);
assert.equal(vm.runInContext("engineLabel({engine_name:'llama.cpp'})",context),'llama.cpp');
click('[data-historymodel]',{historymodel:'a & b'});await settled();
click('[data-statsreset]');await settled();
assert.ok(calls.some(c=>c.path==='/api/stats/reset'));
assert.doesNotMatch(view.innerHTML,/RECENT RUNS/);
summary.per_model.push({id:'vllm-only',has_history:false,tokens:0});
await vm.runInContext('loadStats(true)',context);
assert.doesNotMatch(view.innerHTML,/data-historymodel="vllm-only"/);
console.log('Stats UI expansion, limits, tooltips, config, command and Reset passed');
