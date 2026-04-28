/**
 * Murf AI Suite — Frontend Logic
 * ═══════════════════════════════════════════════════════════════════════════
 * Handles: tab switching, file upload, API calls, chat, audio playback,
 *          dropdowns, loading states, and toast notifications.
 */

const API = "";  // same-origin; change to full URL if deployed separately

// ── DOM refs ───────────────────────────────────────────────────────────────
const $  = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// Tabs
const tabBtns     = $$(".tab-btn");
const tabContents = $$(".tab-content");

// File tab
const dropZone      = $("#drop-zone");
const fileInput     = $("#file-input");
const fileNameEl    = $("#file-name");
const fileTargetDD  = $("#file-target-lang");
const fileVoiceDD   = $("#file-voice");
const fileSubmit    = $("#file-submit");
const fileResult    = $("#file-result");
const fileText      = $("#file-translated-text");
const fileAudio     = $("#file-audio");
const fileSegCount  = $("#file-seg-count");

// Chat tab
const chatTargetDD  = $("#chat-target-lang");
const chatVoiceDD   = $("#chat-voice");
const chatMessages  = $("#chat-messages");
const chatEmpty     = $("#chat-empty");
const chatInput     = $("#chat-input");
const chatSend      = $("#chat-send");
const chatAudio     = $("#chat-audio-player");

// Toast
const toastEl = $("#toast");

// ── State ──────────────────────────────────────────────────────────────────
let selectedFile = null;
let chatBusy     = false;

// ═══════════════════════════════════════════════════════════════════════════
//  INIT — fetch voices & populate dropdowns
// ═══════════════════════════════════════════════════════════════════════════
async function init() {
  try {
    const res  = await fetch(`${API}/api/voices`);
    const data = await res.json();

    populateDropdown(fileTargetDD, Object.keys(data.locales));
    populateDropdown(chatTargetDD, Object.keys(data.locales));
    populateDropdown(fileVoiceDD,  Object.keys(data.voices));
    populateDropdown(chatVoiceDD,  Object.keys(data.voices));

    // Set sensible defaults
    fileTargetDD.value = "Hindi";
    chatTargetDD.value = "Hindi";
    fileVoiceDD.value  = "Hindi - Amit (M)";
    chatVoiceDD.value  = "Hindi - Amit (M)";
  } catch (e) {
    console.warn("Could not fetch voices:", e);
    // Provide fallback options if server isn't running
    const fallbackLocales = ["Hindi", "English US", "French", "German"];
    const fallbackVoices  = [
      "Hindi - Amit (M)", "Hindi - Ayushi (F)",
      "English - Ken (M)", "English - Natalie",
      "French - Adélie", "French - Justine",
      "German - Ruby", "German - Lia"
    ];
    populateDropdown(fileTargetDD, fallbackLocales);
    populateDropdown(chatTargetDD, fallbackLocales);
    populateDropdown(fileVoiceDD,  fallbackVoices);
    populateDropdown(chatVoiceDD,  fallbackVoices);
  }
}

function populateDropdown(sel, items) {
  sel.innerHTML = "";
  items.forEach(item => {
    const opt = document.createElement("option");
    opt.value = item;
    opt.textContent = item;
    sel.appendChild(opt);
  });
}

// ═══════════════════════════════════════════════════════════════════════════
//  TABS
// ═══════════════════════════════════════════════════════════════════════════
tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    tabBtns.forEach(b => b.classList.remove("active"));
    tabContents.forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    $(`#${btn.dataset.tab}`).classList.add("active");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
//  FILE UPLOAD — drag-and-drop + click
// ═══════════════════════════════════════════════════════════════════════════
fileInput.addEventListener("change", (e) => {
  handleFile(e.target.files[0]);
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file) return;
  if (!file.name.endsWith(".txt")) {
    toast("Please upload a .txt file", "error");
    return;
  }
  selectedFile = file;
  fileNameEl.textContent = `📎 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  fileSubmit.disabled = false;
  fileResult.classList.remove("visible");
}

// ═══════════════════════════════════════════════════════════════════════════
//  FILE → SPEECH submit
// ═══════════════════════════════════════════════════════════════════════════
fileSubmit.addEventListener("click", async () => {
  if (!selectedFile) return;

  setLoading(fileSubmit, true);
  fileResult.classList.remove("visible");

  const form = new FormData();
  form.append("file", selectedFile);
  form.append("target_lang", fileTargetDD.value);
  form.append("voice", fileVoiceDD.value);

  try {
    const res  = await fetch(`${API}/api/translate-file`, { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || "Unknown error");

    fileText.textContent = data.translated_text;
    fileSegCount.textContent = `${data.segments} segments`;

    // Build audio from base64
    const audioBlob = base64ToBlob(data.audio_base64, data.audio_mime);
    fileAudio.src = URL.createObjectURL(audioBlob);

    fileResult.classList.add("visible");
    toast("Translation complete!", "success");
  } catch (e) {
    toast(`Error: ${e.message}`, "error");
  } finally {
    setLoading(fileSubmit, false);
  }
});

// ═══════════════════════════════════════════════════════════════════════════
//  CHAT
// ═══════════════════════════════════════════════════════════════════════════
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
});
chatSend.addEventListener("click", sendChat);

async function sendChat() {
  const msg = chatInput.value.trim();
  if (!msg || chatBusy) return;

  chatBusy = true;
  chatInput.value = "";
  setLoading(chatSend, true);
  chatEmpty.style.display = "none";

  // Add user bubble
  addBubble("user", msg);

  try {
    const res = await fetch(`${API}/api/translate-chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message:     msg,
        target_lang: chatTargetDD.value,
        voice:       chatVoiceDD.value,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Unknown error");

    addBubble("bot", data.translated, data.audio_url);
  } catch (e) {
    addBubble("bot", `❌ ${e.message}`);
    toast(`Error: ${e.message}`, "error");
  } finally {
    chatBusy = false;
    setLoading(chatSend, false);
    chatInput.focus();
  }
}

function addBubble(type, text, audioUrl) {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${type}`;

  const label = document.createElement("div");
  label.className = "bubble-label";
  label.textContent = type === "user" ? "You" : "Translation";
  bubble.appendChild(label);

  const content = document.createElement("div");
  content.textContent = text;
  bubble.appendChild(content);

  if (type === "bot" && audioUrl) {
    const playBtn = document.createElement("span");
    playBtn.className = "audio-link";
    playBtn.innerHTML = "🔊 Play Audio";
    playBtn.addEventListener("click", () => {
      chatAudio.src = audioUrl;
      chatAudio.play();
    });
    bubble.appendChild(playBtn);
  }

  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ═══════════════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════════════
function setLoading(btn, on) {
  btn.disabled = on;
  btn.classList.toggle("loading", on);
}

function base64ToBlob(b64, mime) {
  const bytes = atob(b64);
  const arr   = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

let toastTimer;
function toast(msg, type = "success") {
  toastEl.textContent = msg;
  toastEl.className = `toast ${type} show`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 4000);
}

// Navbar scroll effect
window.addEventListener("scroll", () => {
  const nav = $("#navbar");
  nav.style.background = window.scrollY > 40
    ? "rgba(10,10,18,.92)"
    : "rgba(10,10,18,.75)";
});

// ── Boot ───────────────────────────────────────────────────────────────────
init();
