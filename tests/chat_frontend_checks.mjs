import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const base = new URL('../web/js/', import.meta.url);
const protocol = await import('data:text/javascript;base64,' + fs.readFileSync(new URL('chat-protocol.js', base)).toString('base64'));
const {makeRequest, createSSEParser, requestMetrics, streamCompletion, logSince} = protocol;
const frames = [];
const parser = createSSEParser(frame => frames.push(frame));
for (const char of ': comment\r\nevent: custom\r\ndata: one\r\ndata: two\r\n\r\ndata: [DONE]\n\n') parser.push(char);
assert.deepEqual(frames, [{event:'custom',data:'one\ntwo'}, {event:'message',data:'[DONE]'}]);
const defaults = makeRequest('m', [{role:'user',content:'hello'}], '');
assert.deepEqual(Object.keys(defaults).sort(), ['messages','model','stream','stream_options']);
assert.equal(makeRequest('m', [], 'system', {temperature:'0', seed:'-1'}).temperature, 0);
assert.throws(() => makeRequest('m', [], '', {max_tokens:'1.2'}));
assert.throws(() => makeRequest('m', [], '', {top_p:'2'}));
assert.equal(requestMetrics({}).decode, null);
assert.equal(requestMetrics({timings:{draft_n:10,draft_n_accepted:7}}).acceptance, 70);
assert.equal(logSince('a\nb\nc', 'b\nc\nd'), 'd');
assert.match(logSince('old', 'new'), /rotated/);
const sample = 'event: llamaforge.diagnostics\ndata: {"request":{"messages":[{"role":"system","content":"wiki"}]}}\n\n' +
 'data: {"model":"actual","id":"reply-1","choices":[{"delta":{"role":"assistant"}}]}\n\n' +
 'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\n' +
 'data: {"choices":[{"delta":{"content":"こんにちは <script>"}}]}\n\n' +
 'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"prompt_per_second":154.3,"predicted_per_second":16.1,"draft_n":10,"draft_n_accepted":7}}\n\n' +
 'data: {"choices":[],"usage":{"prompt_tokens":1824,"completion_tokens":512}}\n\ndata: [DONE]\n\n';
const bytes = new TextEncoder().encode(sample);
function response(textBytes = bytes) {
 return new Response(new ReadableStream({start(c) {for (const b of textBytes) c.enqueue(Uint8Array.of(b)); c.close();}}),
 {headers:{'Content-Type':'text/event-stream'}});
}
let clock = 0;
const output = await streamCompletion(defaults, {fetchImpl:async () => response(), now:() => clock += 10});
assert.equal(output.content, 'こんにちは <script>');
assert.equal(output.reasoning, 'think');
assert.equal(output.model, 'actual');
assert.equal(output.ttft, 10, 'role/diagnostic frames are not first tokens');
assert.equal(output.usage.completion_tokens, 512);
assert.equal(requestMetrics(output).decode, 16.1);
assert.equal(output.effective_request.messages[0].content, 'wiki');
await assert.rejects(streamCompletion(defaults, {fetchImpl:async () => response(new TextEncoder().encode('data: {"choices":[]}\n\n'))}), /before \[DONE\]/);
await assert.rejects(streamCompletion(defaults, {fetchImpl:async () => new Response('offline', {status:503})}), /503/);
await assert.rejects(streamCompletion(defaults, {fetchImpl:async () => response(new TextEncoder().encode('data: {"error":{"message":"bad model"}}\n\n'))}), /bad model/);

function fixture(empty = false) {
 const nodes = {}, requests = [], calls = [];
 let loaded = false, mode = 'ok';
 const node = id => nodes[id] ||= {value:'',checked:false,disabled:false,hidden:false,dataset:{},style:{},scrollHeight:300,scrollTop:0,clientHeight:300,focus(){}};
 const $ = id => node(id);
 const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const context = vm.createContext({$,esc,setHTML:(el, html)=>{
   el.innerHTML = html;
   for (const match of html.matchAll(/id="([^"]+)"/g)) node('#'+match[1]);
 }, api:async (path,body) => {
   calls.push({path,body});
   if (path.startsWith('/api/chat/diagnostics')) return {models:empty ? [] : [{id:'m',backend:'llamacpp',status:loaded?'loaded':'unloaded'}], model:'m',backend:'llamacpp',engine:'llamacpp',available:!empty,status:loaded?'loaded':'unloaded',router_up:true,port:60687,argv:loaded?['server','--ctx-size','8192']:[],argv_source:'runtime',context_capacity:loaded?8192:null};
   if (path === '/api/models/load') {loaded = true; return {ok:true};}
   if (path.includes('/log')) return {log:'line\nwarning example\nerror example'};
   throw Error(path);
 }, S:{STATE:{config:{}}}, ...protocol, AbortController, DOMException,
 streamCompletion:(body, options) => {
   requests.push(body);
   return streamCompletion(body, {...options, fetchImpl:async (_path, req) => {
     if (mode === 'error') return new Response('router stopped', {status:503});
     if (mode === 'stop') return new Response(new ReadableStream({start(c) {
       c.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'));
       req.signal.addEventListener('abort', () => c.error(new DOMException('Stopped','AbortError')));
     }}), {headers:{'Content-Type':'text/event-stream'}});
     return response();
   }});
 }});
 const source = fs.readFileSync(new URL('test-chat.js', base),'utf8').replace(/^import .*;\r?$/gm,'').replace(/^export /gm,'');
 vm.runInContext(source,context);
 return {nodes,requests,calls,context, mode:v=>{mode=v;}, run:code=>vm.runInContext(code,context)};
}
const ui = fixture();
await ui.run('loadTestChat()');
assert.equal(ui.nodes['#chat-send'].disabled,true);
await ui.nodes['#chat-load'].onclick();
assert.equal(ui.nodes['#chat-send'].disabled,false);
ui.nodes['#chat-input'].value='hello';
await ui.run('sendMessage()');
assert.match(ui.nodes['#chat-messages'].innerHTML,/&lt;script&gt;/);
assert.match(ui.nodes['#chat-last'].innerHTML,/16.1/);
assert.match(ui.nodes['#chat-argv'].textContent,/8192/);
assert.equal(ui.requests[0].messages.length,1);
ui.nodes['#chat-input'].value='follow up';
await ui.run('sendMessage()');
assert.equal(ui.requests[1].messages.length,3);
assert.equal(ui.requests[1].messages[1].role,'assistant');
ui.mode('stop'); ui.nodes['#chat-input'].value='stop test';
const pending = ui.run('sendMessage()');
for (let i=0;i<30;i++) {await new Promise(resolve=>setTimeout(resolve,1)); if (ui.nodes['#chat-messages'].innerHTML.includes('partial')) break;}
ui.nodes['#chat-stop'].onclick();
await pending;
assert.match(ui.nodes['#chat-error'].textContent,/Stopped by user/);
assert.equal(ui.nodes['#chat-send'].disabled,false);
ui.nodes['#chat-clear'].onclick();
assert.equal(ui.run('history.length'),0);
assert.equal(ui.nodes['#chat-last'].textContent,'No request yet.');
ui.mode('error');ui.nodes['#chat-input'].value='error test';await ui.run('sendMessage()');
assert.match(ui.nodes['#chat-error'].textContent,/503/);
assert.equal(ui.nodes['#chat-send'].disabled,false);
const empty = fixture(true);await empty.run('loadTestChat()');
assert.match(empty.nodes['#chat-state'].textContent,/No model/);
assert.equal(empty.nodes['#chat-send'].disabled,true);
let prevented = false;
empty.nodes['#chat-input'].onkeydown({key:'Enter',shiftKey:true,preventDefault:()=>{prevented=true;}});
assert.equal(prevented,false);
console.log('Protocol and UI checks passed');
