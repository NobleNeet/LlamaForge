---
title: Test Chat
section: guides
order: 4
---

# Test Chat

**Test Chat** is a small inference diagnostics screen in the sidebar. Select a model, load it with the existing model controls, and send a message through the panel's existing `/v1/chat/completions` proxy and llama-family router. It supports streaming, **Stop**, **Clear Chat**, Enter to send, and Shift+Enter for a new line. Conversation state lives only in this browser page's memory. Switching the selected model clears the conversation.

If the router is down, **Start router & load** uses the existing router restart API, waits for it to answer, then requests the model load. Loading and errors appear in the screen; generation becomes available after the router reports the model loaded. Model defaults sends no sampling overrides. **Override for this chat** enables optional Temperature, Top P, Top K, Max tokens, and Seed fields; blank fields still use server defaults. The optional system prompt changes only this conversation. Completed messages use Help's existing escaped Markdown renderer; embedded images are omitted.

## Diagnostics

- **Overview** reads current model/backend status from the registry and router. Child port and launch arguments come from a loaded model's `/models` `status.args`. PID is resolved from that child port when supported by the host. Runtime context capacity comes from `/props`, never from saved configuration. Overview is a current observation, not proof that a later request used the same process after a reload.
- **Last Request** shows the requested model and the response's own model/ID, prompt and completion token counts from `usage`, and prompt/decode speeds directly from server `timings`. Unlike Stats, these are per-request values; aggregate or recent throughput is never substituted. Missing values remain **Unavailable**, including builds that do not include timings in OpenAI streams. Speculative draft/accepted counts and acceptance rate appear only when supplied by the response.
- **TTFT** measures browser elapsed time from sending the completion HTTP request to the first nonempty text, reasoning, or tool output. It includes network/queue/prefill time. Role-only chunks and diagnostic events do not count. **Total** measures until the stream finishes or fails. These are client observations, not additional server throughput estimates. The assistant footer repeats the available generation metrics.
- **Parameters** shows the router's runtime argv, preserving argument boundaries and omitting authentication values. Unloaded models' planned settings are not presented as running arguments. CPU/GPU acceleration is not inferred from saved build flags.
- **Logs** reuses the router/stdout log APIs. All, Request, Warnings, and Errors filters are available. Request is the shared log window since send; concurrent clients may appear, and rotation/truncation is indicated when overlap cannot be recovered. This is not exact request-ID correlation.
- **API Details** includes the browser's Raw Request, the Effective Request after the normal Context Wiki injection, and assembled response metadata. Effective Request is delivered as an opt-in `llamaforge.diagnostics` SSE event (`X-LlamaForge-Diagnostics: 1`); ordinary clients retain their existing event stream. No token breakdown by system/wiki/conversation is estimated.

**Stop** aborts the browser fetch. The proxy detects a disconnected client, closes the upstream connection, and releases its request tracking, including while waiting for streamed output. Cancellation still depends on upstream disconnect handling; an upstream connection that has not yet returned HTTP headers cannot be interrupted by the relay until it returns. Interrupted output is visibly marked and remains available as partial conversation text.

## Supported inference path

The existing panel completion proxy targets the active llama-family router: llama.cpp, or a router-capable ik_llama build. vLLM remains visible as a registry backend, but Test Chat does not send its models through the llama.cpp proxy; the screen explains this limitation. Existing vLLM clients and APIs are unchanged.

This screen does not persist chats, add RAG, attachments, tools, audio, exports, or another inference engine. Build's scheduled maintenance returns HTTP 503 normally; Test Chat displays the error without breaking the page.

The runtime fields and timings follow upstream [router model metadata](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-models.cpp) and [server timing statistics](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-common.cpp).
