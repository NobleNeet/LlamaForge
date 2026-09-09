// Test Chat transport uses the same OpenAI proxy as external clients.
// No aggregate Stats rates are attributed to individual requests.
export function createSSEParser(onEvent) {
  let buffer = '', data = [], event = 'message';
  function line(value) {
    if (value === '') {
      if (data.length) onEvent({event, data: data.join('\n')});
      data = []; event = 'message';
    } else if (value.startsWith('data:')) data.push(value.slice(5).replace(/^ /, ''));
    else if (value.startsWith('event:')) event = value.slice(6).trim();
  }
  return {
    push(text) {
      buffer += text;
      let i;
      while ((i = buffer.indexOf('\n')) >= 0) {
        line(buffer.slice(0, i).replace(/\r$/, ''));
        buffer = buffer.slice(i + 1);
      }
    },
    finish() { if (buffer) line(buffer.replace(/\r$/, '')); line(''); buffer = ''; },
  };
}

export function makeRequest(model, history, system, override = null) {
  const messages = history.filter(m => m.content && (m.role === 'user' || m.role === 'assistant'))
    .map(({role, content}) => ({role, content}));
  if (system.trim()) messages.unshift({role: 'system', content: system});
  const body = {model, messages, stream: true, stream_options: {include_usage: true}};
  if (override) {
    const bounds = {temperature:[0, 2], top_p:[0, 1], top_k:[0, Infinity], max_tokens:[1, Infinity], seed:[-1, Infinity]};
    for (const [key, [min, max]] of Object.entries(bounds)) {
      const value = override[key];
      if (value === '' || value == null) continue;
      const n = Number(value);
      if (!Number.isFinite(n) || n < min || n > max ||
          (['top_k', 'max_tokens', 'seed'].includes(key) && !Number.isSafeInteger(n))) {
        throw new Error(`Invalid ${key}`);
      }
      body[key] = n;
    }
  }
  return body;
}

export function requestMetrics(result) {
  const u = result.usage || {}, t = result.timings || {};
  const numeric = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
  const drafted = numeric(t.draft_n), accepted = numeric(t.draft_n_accepted);
  return {
    prompt: numeric(u.prompt_tokens), generated: numeric(u.completion_tokens),
    prefill: numeric(t.prompt_per_second), decode: numeric(t.predicted_per_second),
    ttft: numeric(result.ttft), total: numeric(result.total),
    drafted, accepted, acceptance: drafted > 0 && accepted !== null ? 100 * accepted / drafted : null,
  };
}

export async function streamCompletion(body, {signal, token = '', onUpdate = () => {},
    fetchImpl = fetch, now = () => performance.now()} = {}) {
  const start = now();
  const result = {content:'', reasoning:'', usage:null, timings:null, ttft:null,
    total:null, finish_reason:null, model:null, id:null, effective_request:null, tool_calls:[]};
  let done = false, reader;
  const parser = createSSEParser(frame => {
    if (frame.data === '[DONE]') { done = true; return; }
    const chunk = JSON.parse(frame.data);
    if (frame.event === 'llamaforge.diagnostics') {
      result.effective_request = chunk.request;
      onUpdate(result); return;
    }
    if (chunk.error) throw new Error(typeof chunk.error === 'string' ? chunk.error : chunk.error.message || JSON.stringify(chunk.error));
    if (chunk.model) result.model = chunk.model;
    if (chunk.id) result.id = chunk.id;
    if (chunk.usage && Object.keys(chunk.usage).length) result.usage = chunk.usage;
    if (chunk.timings) result.timings = chunk.timings;
    for (const choice of chunk.choices || []) {
      if (choice.index != null && choice.index !== 0) continue;
      const delta = choice.delta || {};
      const text = typeof delta.content === 'string' ? delta.content : '';
      const reasoning = delta.reasoning_content || delta.reasoning || '';
      if ((text || reasoning || delta.tool_calls?.length) && result.ttft === null) result.ttft = now() - start;
      result.content += text;
      if (typeof reasoning === 'string') result.reasoning += reasoning;
      if (delta.tool_calls) result.tool_calls.push(...delta.tool_calls);
      if (choice.finish_reason != null) result.finish_reason = choice.finish_reason;
      if (choice.timings) result.timings = choice.timings;
    }
    onUpdate(result);
  });
  try {
    const headers = {'Content-Type':'application/json', 'X-LlamaForge-Diagnostics':'1'};
    if (token) headers.Authorization = 'Bearer ' + token;
    const response = await fetchImpl('/v1/chat/completions', {method:'POST', headers,
      body:JSON.stringify(body), signal});
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`HTTP ${response.status}: ${text.slice(0, 2000)}`);
    }
    if (!response.body) throw new Error('Streaming response is unavailable');
    if (!(response.headers.get('content-type') || '').includes('text/event-stream')) {
      throw new Error('Expected an SSE response from the Chat Completion endpoint');
    }
    reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (!done) {
      const next = await reader.read();
      if (next.done) { parser.push(decoder.decode()); parser.finish(); break; }
      parser.push(decoder.decode(next.value, {stream:true}));
    }
    if (!done) throw new Error('Stream ended before [DONE]; response may be incomplete');
    return result;
  } finally {
    result.total = now() - start;
    onUpdate(result);
    if (reader) { try { await reader.cancel(); } catch (_) {} reader.releaseLock(); }
  }
}

// A tail is shared by all clients. This is only a time window, never request attribution.
export function logSince(baseline, current) {
  if (!baseline) return current;
  if (current.startsWith(baseline)) return current.slice(baseline.length);
  const old = baseline.split('\n'), lines = current.split('\n');
  for (let n = Math.min(old.length, lines.length); n > 0; n--) {
    if (old.slice(-n).join('\n') === lines.slice(0, n).join('\n')) return lines.slice(n).join('\n');
  }
  return '[Log rotated or previous lines fell outside the tail]\n' + current;
}
