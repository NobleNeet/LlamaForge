"""Run the real Build UI handlers with a small DOM/API fixture in Node."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which('node'), 'Node.js required')
class CustomBuildUiTests(unittest.TestCase):
    def test_select_validate_save_build_edit_remove(self):
        script = r'''
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const source = fs.readFileSync(JSON.parse(fs.readFileSync(0, 'utf8')), 'utf8')
 .replace(/^import .*;\r?$/gm, '').replace(/^export /gm, '');
const nodes = {}, calls = [], dialogs = [], confirmations = [];
let selected = 'custom-one';
let custom = {id:selected,name:'My fork',repository:'https://example.com/fork',branch:'main',source:'/src',build:'/build',server_binary:'{build}/server',build_command:'echo hello'};
const builtins = [{id:'llamacpp',name:'llama.cpp',builtin:true},{id:'ikllama',name:'ik_llama',builtin:true}];
function dialog() {
 const children = {'form':{reportValidity:()=>true}};
 return {querySelector:s=>children[s] ||= {}, querySelectorAll:()=>[], showModal(){}, close(){this.onclose();}, remove(){this.removed=true;}, children};
}
const context = vm.createContext({
 $: s => nodes[s] ||= {style:{}}, esc: s=>String(s), setHTML:(n,h)=>n.innerHTML=h,
 localStorage:{getItem:()=>selected,setItem:(k,v)=>selected=v},
 setInterval:()=>1, clearInterval:()=>{}, toast:()=>{}, agoText:()=>'', fmtDur:()=>'',
 confirm: text=>{confirmations.push(text);return true;},
 document:{createElement:()=>{const d=dialog();dialogs.push(d);return d;},body:{appendChild(){}}},
 FormData: class { constructor(form){ return Object.entries({...custom,id:undefined}).filter(([k])=>k!=='id')[Symbol.iterator](); } },
 api:async(path,body)=>{
  calls.push({path,body});
  if(path==='/api/build/targets')return {targets:custom?[...builtins,{...custom,builtin:false}]:builtins};
  if(path.startsWith('/api/build/info'))return {};
  if(path==='/api/state')return {active_engine:'llamacpp',config:{}};
  if(path.startsWith('/api/vllm/version'))return {error:'unsupported'};
  if(path.startsWith('/api/build/log'))return {};
  if(path==='/api/build/targets/validate')return {ok:true,server_binary:'/build/server'};
  if(path==='/api/build/targets/save'){ custom={...body,id:body.id||'custom-new'};return {ok:true,target:custom}; }
  if(path==='/api/build/targets/remove'){custom=null;return {ok:true};}
  if(path==='/api/build/start')return {started:true};
  throw Error(path);
 }
});
vm.runInContext(source,context);
(async()=>{
 await vm.runInContext('loadBuild()',context);
 let html=nodes['#view-build'].innerHTML;
 assert.match(html,/id="build-target"/);assert.match(html,/My fork/);assert.match(html,/ik_llama/);
 assert.doesNotMatch(html,/Acceleration Backend/);assert.doesNotMatch(html,/id="btn-switch-engine"/);
 delete nodes['#opt-pull'];
 // Return null for an absent checkbox, as a real DOM would.
 const select=context.$;context.$=s=>s==='#opt-pull'?null:select(s);
 await nodes['#btn-build'].onclick();
 assert.equal(JSON.stringify(calls.find(x=>x.path==='/api/build/start').body),JSON.stringify({pull:true,target:'custom-one'}));
 nodes['#btn-edit-target'].onclick();
 let d=dialogs.at(-1);assert.match(d.innerHTML,/textarea/);assert.match(d.innerHTML,/echo hello/);
 await d.children['[data-action="validate"]'].onclick();
 assert.match(d.children['[data-status]'].textContent,/Valid/);
 assert.equal(calls.filter(x=>x.path==='/api/build/start').length,1);
 await d.children.form.onsubmit({preventDefault(){}});
 // Submit handler deliberately launches an async operation; drain its awaits.
 for(let i=0;i<20;i++)await Promise.resolve();
 assert.equal(calls.find(x=>x.path==='/api/build/targets/save').body.id,'custom-one');
 assert.equal(d.removed,true);
 nodes['#btn-add-target'].onclick();d=dialogs.at(-1);
 await d.children.form.onsubmit({preventDefault(){}});
 for(let i=0;i<20;i++)await Promise.resolve();
 assert.equal(selected,'custom-new');
 assert.equal(calls.filter(x=>x.path==='/api/build/targets/save').at(-1).body.id,undefined);
 await nodes['#btn-remove-target'].onclick();
 for(let i=0;i<20;i++)await Promise.resolve();
 assert.match(confirmations[0],/remain on disk/);
 assert.equal(selected,'llamacpp');
 assert.match(nodes['#view-build'].innerHTML,/Acceleration Backend/);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
        result = subprocess.run([shutil.which('node'), '-e', script],
                                input=json.dumps(str(Path(__file__).resolve().parents[1] / 'web/js/build.js')),
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
