const API_URL =
  window.location.origin === "null" ||
  window.location.origin.startsWith("file:")
    ? "http://127.0.0.1:8000"
    : window.location.origin;

const $ = (id) => document.getElementById(id);
const elements = {
  sidebar: $("sidebar"),
  sessionList: $("sessionList"),
  newChatBtn: $("newChatBtn"),
  messages: $("messages"),
  chatTitle: $("chatTitle"),
  chatSubtitle: $("chatSubtitle"),
  composerForm: $("composerForm"),
  messageInput: $("messageInput"),
  sendBtn: $("sendBtn"),
  attachBtn: $("attachBtn"),
  fileInput: $("fileInput"),
  attachRow: $("attachRow"),
  attachChip: $("attachChip"),
  micBtn: $("micBtn"),
  useUploads: $("useUploads"),
  useBase: $("useBase"),
  pipelineStrip: $("pipelineStrip"),
  pipelineSteps: $("pipelineSteps"),
  inspStatus: $("inspStatus"),
  inspLatency: $("inspLatency"),
  inspAnalysis: $("inspAnalysis"),
  inspSources: $("inspSources"),
  baseIndexStat: $("baseIndexStat"),
  uploadIndexStat: $("uploadIndexStat"),
  healthStat: $("healthStat"),
};

const allowedExtensions = new Set([
  ".pdf",
  ".txt",
  ".md",
  ".markdown",
  ".csv",
  ".json",
  ".docx",
  ".log",
]);
const MAX_FILE_BYTES = 25 * 1024 * 1024;
const SESSION_STORAGE_KEY = "aura.currentSession";
let currentSessionId = localStorage.getItem(SESSION_STORAGE_KEY);
let busy = false;
let uploading = false;
let recorder = null;
let recorderStream = null;
let generationMode = "extractive";

async function requestJson(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, options);
  } catch (error) {
    throw new Error(`Cannot reach AURA backend at ${API_URL}. Start the server and try again.`);
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error("AURA returned an invalid response.");
  }
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed (${response.status}).`);
  }
  return payload;
}

function addText(parent, tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = text || "";
  parent.appendChild(element);
  return element;
}

function makeBadge(text, tone = "") {
  const badge = document.createElement("span");
  badge.className = `badge ${tone}`.trim();
  badge.textContent = text;
  return badge;
}

function formatTime(timestamp) {
  if (!timestamp) return "Saved chat";
  return new Date(timestamp * 1000).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function renderWelcome() {
  const hero = document.createElement("div");
  hero.className = "empty-hero";
  addText(hero, "h2", "", "What can I help you explore?");
  addText(
    hero,
    "p",
    "",
    "Ask a question about your documents or the indexed AURA knowledge base."
  );
  const suggestions = document.createElement("div");
  suggestions.className = "suggestions";
  [
    "Summarize the key points in my files",
    "What evidence supports the main conclusion?",
    "Compare the approaches described in my documents",
  ].forEach((prompt) => {
    const button = addText(suggestions, "button", "", prompt);
    button.type = "button";
    button.dataset.prompt = prompt;
  });
  hero.appendChild(suggestions);
  elements.messages.appendChild(hero);
}

function renderSources(container, sources) {
  container.replaceChildren();
  if (!Array.isArray(sources) || sources.length === 0) {
    container.classList.add("empty");
    addText(container, "p", "", "No retrieved sources for this answer.");
    return;
  }

  container.classList.remove("empty");
  sources.slice(0, 5).forEach((source, index) => {
    const card = document.createElement("article");
    card.className = "source-card";
    const sourceName =
      source.source_file ||
      source.origin ||
      source.query ||
      `Source ${index + 1}`;
    const header = document.createElement("div");
    header.className = "source-meta";
    addText(header, "strong", "", sourceName);
    const score = Number(source.final_score ?? source.similarity ?? 0);
    addText(header, "span", "", `Score ${score.toFixed(2)}`);
    card.appendChild(header);
    addText(card, "p", "source-text", source.text || "Evidence unavailable.");
    container.appendChild(card);
  });
}

function renderPipeline(pipeline = []) {
  elements.pipelineSteps.replaceChildren();
  if (!pipeline.length) {
    elements.pipelineStrip.hidden = true;
    return;
  }

  elements.pipelineStrip.hidden = false;
  pipeline.forEach((stage) => {
    const item = document.createElement("span");
    item.className = `pipe-step ${stage.status || ""}`.trim();
    const elapsed =
      Number(stage.ms) > 0 ? ` · ${Number(stage.ms).toFixed(1)} ms` : "";
    item.textContent = `${stage.name}${elapsed}`;
    item.title = stage.description || stage.name;
    elements.pipelineSteps.appendChild(item);
  });
}

function renderAnalysis(analysis = {}) {
  elements.inspAnalysis.replaceChildren();
  const values = [analysis.language, analysis.query_type, analysis.complexity]
    .filter(Boolean);
  if (!values.length) {
    addText(elements.inspAnalysis, "span", "", "No analysis yet");
    return;
  }
  values.forEach((value) => addText(elements.inspAnalysis, "span", "", value));
}

function renderMessage(message) {
  const role = ["user", "assistant", "system"].includes(message.role)
    ? message.role
    : "system";
  const bubble = document.createElement("article");
  bubble.className = `msg ${role}`;
  addText(bubble, "div", "message-content", message.content || "");

  const meta = message.meta || {};
  if (role === "assistant" && (meta.grounded !== undefined || meta.total_ms)) {
    const badges = document.createElement("div");
    badges.className = "meta";
    badges.appendChild(
      makeBadge(
        meta.grounded ? "Grounded" : "Needs more evidence",
        meta.grounded ? "ok" : "warn"
      )
    );
    if (meta.total_ms) {
      badges.appendChild(makeBadge(`${Number(meta.total_ms).toFixed(1)} ms`));
    }
    if (meta.generation_mode && meta.generation_mode !== "not_run") {
      badges.appendChild(makeBadge(meta.generation_mode));
    }
    bubble.appendChild(badges);
  }
  if (role === "system" && meta.type === "upload" && meta.filename) {
    const details = document.createElement("div");
    details.className = "meta";
    details.appendChild(makeBadge(`${meta.chunks || 0} indexed chunks`, "ok"));
    bubble.appendChild(details);
  }
  elements.messages.appendChild(bubble);
}

function renderSession(session) {
  currentSessionId = session.session_id;
  localStorage.setItem(SESSION_STORAGE_KEY, currentSessionId);
  elements.chatTitle.textContent = session.title || "AURA Chat";
  const generationLabel =
    generationMode === "ollama" ? "Local model" : "Fast extractive mode";
  elements.chatSubtitle.textContent = `${(session.attachments || []).length} attached file(s) · ${generationLabel}`;
  elements.messages.replaceChildren();
  const history = session.messages || [];
  if (!history.length) {
    renderWelcome();
    elements.messages.scrollTop = 0;
  } else {
    history.forEach(renderMessage);
    scrollToLatest();
  }

  const lastAnswer = [...history].reverse().find((item) => item.role === "assistant");
  if (lastAnswer) {
    const meta = lastAnswer.meta || {};
    elements.inspStatus.textContent = meta.grounded ? "Grounded" : "Abstained";
    elements.inspLatency.textContent = meta.total_ms
      ? `${Number(meta.total_ms).toFixed(1)} ms`
      : "—";
    renderAnalysis(meta.analysis || {});
    renderSources(elements.inspSources, meta.sources || []);
    renderPipeline(meta.pipeline || []);
  } else {
    elements.inspStatus.textContent = "Waiting";
    elements.inspLatency.textContent = "—";
    renderAnalysis();
    renderSources(elements.inspSources, []);
    renderPipeline();
  }
}

function scrollToLatest() {
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

async function refreshSessionList() {
  const data = await requestJson("/api/sessions");
  elements.sessionList.replaceChildren();
  if (!data.sessions.length) {
    addText(elements.sessionList, "p", "session-empty", "Your saved chats will appear here.");
    return;
  }
  data.sessions.forEach((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `session-item ${session.session_id === currentSessionId ? "active" : ""}`;
    addText(button, "span", "", session.title || "New chat");
    addText(
      button,
      "small",
      "",
      `${formatTime(session.updated_at)} · ${session.message_count || 0} messages`
    );
    button.addEventListener("click", () => loadSession(session.session_id));
    elements.sessionList.appendChild(button);
  });
}

async function loadSession(sessionId) {
  const session = await requestJson(`/api/sessions/${encodeURIComponent(sessionId)}`);
  renderSession(session);
  await refreshSessionList();
  elements.sidebar.classList.remove("open");
}

async function createSession() {
  const session = await requestJson("/api/sessions", { method: "POST" });
  renderSession(session);
  await refreshSessionList();
  elements.messageInput.focus();
}

function setBusy(value) {
  busy = value;
  elements.sendBtn.disabled = value || uploading;
  elements.attachBtn.disabled = value || uploading;
  elements.micBtn.disabled = value || uploading;
  elements.messageInput.disabled = value;
}

function showTyping() {
  const bubble = document.createElement("article");
  bubble.className = "msg assistant typing";
  bubble.setAttribute("aria-label", "AURA is working");
  bubble.innerHTML = '<span></span><span></span><span></span>';
  elements.messages.appendChild(bubble);
  scrollToLatest();
  return bubble;
}

async function sendMessage(text = elements.messageInput.value) {
  const message = text.trim();
  if (!message || busy || uploading) return;

  if (!currentSessionId) await createSession();
  elements.messageInput.value = "";
  elements.messageInput.style.height = "auto";
  renderMessage({ role: "user", content: message });
  const typing = showTyping();
  setBusy(true);
  elements.inspStatus.textContent = "Analyzing and retrieving";
  renderPipeline([
    { name: "listen", status: "done" },
    { name: "analyze", status: "active" },
    { name: "retrieve", status: "idle" },
    { name: "rerank", status: "idle" },
    { name: "generate", status: "idle" },
    { name: "guardrail", status: "idle" },
  ]);

  try {
    const result = await requestJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: currentSessionId,
        use_base_corpus: elements.useBase.checked,
        use_uploads: elements.useUploads.checked,
      }),
    });
    typing.remove();
    await loadSession(result.session_id);
    elements.inspStatus.textContent = result.grounded
      ? `Grounded · ${Math.round(Number(result.confidence || 0) * 100)}% confidence`
      : result.reason || "More evidence needed";
    elements.inspLatency.textContent = `${Number(result.total_ms || 0).toFixed(1)} ms`;
    renderAnalysis(result.analysis || {});
    renderSources(elements.inspSources, result.sources || []);
    renderPipeline(result.pipeline || []);
  } catch (error) {
    typing.remove();
    try {
      await loadSession(currentSessionId);
    } catch {
      renderMessage({ role: "system", content: "The message could not be saved to this chat." });
    }
    renderMessage({ role: "system", content: error.message });
    elements.inspStatus.textContent = "Request failed";
    elements.pipelineStrip.hidden = true;
  } finally {
    setBusy(false);
    elements.messageInput.focus();
  }
}

function renderAttachmentStatus(name, status, isError = false) {
  elements.attachRow.hidden = false;
  const chip = document.createElement("span");
  chip.className = `attach-chip ${status === "Indexing…" ? "uploading" : ""}`;
  chip.textContent = `${name} · ${status}`;
  if (isError) chip.classList.add("error");
  elements.attachChip.appendChild(chip);
  return chip;
}

async function uploadFiles(files) {
  if (!files.length || busy || uploading) return;
  uploading = true;
  elements.attachChip.replaceChildren();
  elements.attachRow.hidden = false;
  elements.sendBtn.disabled = true;
  elements.attachBtn.disabled = true;
  try {
    if (!currentSessionId) await createSession();
    for (const file of files) {
      const extension = `.${file.name.split(".").pop()}`.toLowerCase();
      if (!allowedExtensions.has(extension)) {
        renderAttachmentStatus(file.name, "Unsupported file type", true);
        continue;
      }
      if (file.size === 0 || file.size > MAX_FILE_BYTES) {
        renderAttachmentStatus(
          file.name,
          file.size === 0 ? "Empty file" : "Over 25 MB",
          true
        );
        continue;
      }

      const status = renderAttachmentStatus(file.name, "Indexing…");
      const form = new FormData();
      form.append("file", file, file.name);
      form.append("session_id", currentSessionId);
      try {
        await requestJson("/api/upload", { method: "POST", body: form });
        status.textContent = `${file.name} · Indexed`;
        status.classList.remove("uploading");
        await loadSession(currentSessionId);
      } catch (error) {
        status.textContent = `${file.name} · Failed: ${error.message}`;
        status.classList.add("error");
        status.classList.remove("uploading");
      }
    }
    await refreshHealth();
    await refreshSessionList();
  } finally {
    uploading = false;
    elements.sendBtn.disabled = busy;
    elements.attachBtn.disabled = busy;
    elements.fileInput.value = "";
    window.setTimeout(() => {
      elements.attachChip.replaceChildren();
      elements.attachRow.hidden = true;
    }, 7000);
  }
}

async function refreshHealth() {
  try {
    const health = await requestJson("/health", { cache: "no-store" });
    elements.healthStat.textContent = health.aura_ready ? "Ready" : "Degraded";
    elements.healthStat.className = health.aura_ready ? "healthy" : "degraded";
    elements.baseIndexStat.textContent = Number(
      health.base_index_size || 0
    ).toLocaleString();
    elements.uploadIndexStat.textContent = Number(
      health.upload_index_size || 0
    ).toLocaleString();
    generationMode = health.generation_mode || "extractive";
    if (!health.aura_ready) {
      elements.chatSubtitle.textContent =
        health.error || "Backend is starting or the local embedding model is unavailable.";
    }
  } catch (error) {
    elements.healthStat.textContent = "Offline";
    elements.healthStat.className = "degraded";
    elements.chatSubtitle.textContent = error.message;
  }
}

async function toggleRecording() {
  if (recorder?.state === "recording") {
    recorder.stop();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    renderMessage({
      role: "system",
      content: "This browser does not support microphone recording.",
    });
    return;
  }

  try {
    recorderStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(recorderStream);
    const chunks = [];
    recorder.ondataavailable = (event) => {
      if (event.data?.size) chunks.push(event.data);
    };
    recorder.onstop = async () => {
      elements.micBtn.classList.remove("recording");
      recorderStream?.getTracks().forEach((track) => track.stop());
      recorderStream = null;
      if (!chunks.length) {
        setBusy(false);
        return;
      }
      setBusy(true);
      const form = new FormData();
      const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
      form.append("file", blob, "aura-voice.webm");
      if (currentSessionId) form.append("session_id", currentSessionId);
      try {
        const result = await requestJson("/api/voice", { method: "POST", body: form });
        await loadSession(result.session_id);
        elements.inspStatus.textContent = result.grounded
          ? "Grounded voice answer"
          : result.reason || "More evidence needed";
        elements.inspLatency.textContent = `${Number(result.total_ms || 0).toFixed(1)} ms`;
        renderAnalysis(result.analysis || {});
        renderSources(elements.inspSources, result.sources || []);
        renderPipeline(result.pipeline || []);
      } catch (error) {
        renderMessage({ role: "system", content: error.message });
      } finally {
        setBusy(false);
      }
    };
    recorder.start();
    elements.micBtn.classList.add("recording");
    elements.micBtn.title = "Stop recording";
    window.setTimeout(() => {
      if (recorder?.state === "recording") recorder.stop();
    }, 7000);
  } catch (error) {
    recorderStream?.getTracks().forEach((track) => track.stop());
    recorderStream = null;
    renderMessage({ role: "system", content: `Microphone error: ${error.message}` });
  }
}

elements.newChatBtn.addEventListener("click", () => {
  createSession().catch((error) =>
    renderMessage({ role: "system", content: error.message })
  );
});
elements.composerForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage();
});
elements.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composerForm.requestSubmit();
  }
});
elements.messageInput.addEventListener("input", () => {
  elements.messageInput.style.height = "auto";
  elements.messageInput.style.height = `${Math.min(
    elements.messageInput.scrollHeight,
    160
  )}px`;
});
elements.attachBtn.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", (event) =>
  uploadFiles(Array.from(event.target.files || []))
);
elements.micBtn.addEventListener("click", toggleRecording);
$("openSidebar").addEventListener("click", () => elements.sidebar.classList.add("open"));
$("closeSidebar").addEventListener("click", () =>
  elements.sidebar.classList.remove("open")
);

elements.messages.addEventListener("click", (event) => {
  const button = event.target.closest("[data-prompt]");
  if (button) sendMessage(button.dataset.prompt);
});

const composer = document.querySelector(".composer");
["dragenter", "dragover"].forEach((eventName) => {
  composer.addEventListener(eventName, (event) => {
    event.preventDefault();
    composer.classList.add("drag-active");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  composer.addEventListener(eventName, (event) => {
    event.preventDefault();
    composer.classList.remove("drag-active");
  });
});
composer.addEventListener("drop", (event) =>
  uploadFiles(Array.from(event.dataTransfer?.files || []))
);

async function initialize() {
  await refreshHealth();
  try {
    const data = await requestJson("/api/sessions");
    const preferredSession = data.sessions.find(
      (session) => session.session_id === currentSessionId
    );
    if (preferredSession) {
      await loadSession(preferredSession.session_id);
    } else if (data.sessions.length) {
      await loadSession(data.sessions[0].session_id);
    } else {
      await createSession();
    }
  } catch (error) {
    renderWelcome();
    renderMessage({ role: "system", content: error.message });
  }
}

initialize();
window.setInterval(refreshHealth, 30000);
