/* ================================================================
   fracture_agent front-end — opens an SSE stream to the Flask backend and
   dispatches events into the correct panel.
   ================================================================ */
(() => {
  "use strict";

  // ---- element handles --------------------------------------------- //
  const $ = (id) => document.getElementById(id);

  const promptEl   = $("prompt");
  const materialEl = $("material");
  const geometryEl = $("geometry");
  const physicsEl  = $("physics");
  const fractureEl = $("fracture");
  const runBtn     = $("run-btn");
  const examplesBtn = $("examples-btn");
  const historyBtn  = $("history-btn");

  const sessionTag  = $("session-tag");
  const spinner     = $("spinner");
  const phasePill   = $("phase-pill");
  const costMeter   = $("cost-meter");

  const activityLog = $("activity-log");
  const decisionsLog = $("decisions-log");
  const assumptionsLog = $("assumptions-log");
  const assumptionsCount = $("assumptions-count");
  const consoleLog  = $("console-log");
  const consoleClear = $("console-clear");
  const consoleAutoscroll = $("console-autoscroll");
  const filterToggles = $("activity-filters");

  const verdictBox  = $("verdict-box");
  const metricsBox  = $("metrics-box");
  const healthBox   = $("health-box");
  const healthScore = $("health-score");
  const healthVerdict = $("health-verdict");
  const healthFlags = $("health-flags");
  const healthBars  = $("health-bars");

  const imagesCard  = $("images-card");
  const imagesBox   = $("images-box");
  const downloadsCard = $("downloads-card");
  const downloadsBox = $("downloads-box");
  const specCard = $("spec-card");
  const specPre  = $("spec-pre");
  const specToggle = $("spec-toggle");

  const chatCard    = $("chat-card");
  const chatHistory = $("chat-history");
  const chatForm    = $("chat-form");
  const chatInput   = $("chat-input");

  // upload zone
  const uploadZone  = $("upload-zone");
  const uploadInput = $("image-input");
  const uploadEmpty = $("upload-empty");
  const uploadList  = $("upload-list");

  // modals
  const examplesModal = $("examples-modal");
  const examplesClose = $("examples-close");
  const examplesList  = $("examples-list");
  const historyModal  = $("history-modal");
  const historyClose  = $("history-close");
  const historyList   = $("history-list");
  const clarifyModal  = $("clarify-modal");
  const clarifyForm   = $("clarify-form");
  const clarifyQs     = $("clarify-questions");
  const clarifySubmit = $("clarify-submit");
  const clarifySkip   = $("clarify-skip");

  const toast = $("toast");

  // ---- state ------------------------------------------------------- //
  let activeSession = null;
  let sessionDirName = null;
  let seenImages = new Set();
  let hadError = false;
  let uploads = [];
  let activityFilters = new Set(["status", "decision", "error", "result"]);

  // ---- helpers ----------------------------------------------------- //
  function clear(el) { el.innerHTML = ""; }
  function placeholder(el, text) {
    clear(el);
    const d = document.createElement("div");
    d.className = "placeholder";
    d.textContent = text;
    el.appendChild(d);
  }
  function showToast(text, kind = "") {
    toast.className = "toast" + (kind ? " " + kind : "");
    toast.textContent = text;
    toast.classList.remove("hidden");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => toast.classList.add("hidden"), 3500);
  }
  function ts() {
    return new Date().toLocaleTimeString([], {
      hour: "2-digit", minute: "2-digit", second: "2-digit"
    });
  }
  function fmtBytes(n) {
    if (n == null) return "";
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    return (n / (1024 * 1024)).toFixed(2) + " MB";
  }
  function fmtCost(usd) {
    if (usd == null) return "$0.00";
    if (usd < 0.001) return "$" + usd.toFixed(5);
    if (usd < 0.01)  return "$" + usd.toFixed(4);
    return "$" + usd.toFixed(3);
  }

  function appendEntry(container, category, text, opts = {}) {
    const ph = container.querySelector(".placeholder");
    if (ph) ph.remove();
    const entry = document.createElement("div");
    entry.className = `entry ${category}`;
    entry.dataset.category = category;
    if (!activityFilters.has(category) && container === activityLog) {
      entry.style.display = "none";
    }
    const tsEl = document.createElement("span");
    tsEl.className = "ts";
    tsEl.textContent = ts();
    const msg = document.createElement("span");
    msg.className = "msg";
    if (category === "error" && (text.includes("\n") || text.length > 200)) {
      msg.style.whiteSpace = "pre-wrap";
      msg.style.fontFamily = "var(--mono)";
      msg.style.fontSize = "11.5px";
      msg.style.lineHeight = "1.4";
    }
    if (opts.phase) {
      const ph2 = document.createElement("span");
      ph2.className = "phase";
      ph2.textContent = opts.phase;
      msg.appendChild(ph2);
    }
    msg.appendChild(document.createTextNode(text));
    entry.appendChild(tsEl);
    entry.appendChild(msg);
    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
    return entry;
  }

  function appendConsole(line) {
    const ph = consoleLog.querySelector(".placeholder");
    if (ph) ph.remove();
    consoleLog.appendChild(document.createTextNode(line + "\n"));
    if (consoleAutoscroll.checked) {
      consoleLog.scrollTop = consoleLog.scrollHeight;
    }
  }

  // ---- cost meter -------------------------------------------------- //
  function updateCostMeter(totals, iters, active = true) {
    if (!totals) {
      costMeter.classList.remove("active");
      costMeter.innerHTML = '<span class="meter-line"><span class="dot"></span>idle</span>';
      return;
    }
    if (active) costMeter.classList.add("active");
    else costMeter.classList.remove("active");
    const cost = fmtCost(totals.cost_usd);
    const calls = totals.n_llm_calls || 0;
    const ttok = totals.total_tokens || 0;
    let extras = "";
    if (iters) {
      const bits = [];
      if (iters.architect_rounds > 1) bits.push(`arch×${iters.architect_rounds}`);
      if (iters.debugger_attempts) bits.push(`dbg×${iters.debugger_attempts}`);
      if (iters.mesh_rescales) bits.push(`mesh×${iters.mesh_rescales}`);
      if (iters.reflect_revise_cycles) bits.push(`rev×${iters.reflect_revise_cycles}`);
      if (bits.length) extras = `<span class="sep">·</span><span title="iteration counters">${bits.join(" ")}</span>`;
    }
    costMeter.innerHTML = (
      `<span class="meter-line">` +
      `<span class="dot"></span>` +
      `<span class="v" title="LLM cost">${cost}</span>` +
      `<span class="sep">·</span>` +
      `<span title="LLM calls">${calls} calls</span>` +
      `<span class="sep">·</span>` +
      `<span title="Total tokens (prompt + output + thinking)">${(ttok/1000).toFixed(1)}k tok</span>` +
      extras +
      `</span>`
    );
  }

  // ---- metrics ----------------------------------------------------- //
  function showMetrics(summary) {
    clear(metricsBox);
    if (!summary) return;
    const items = [
      ["cracked",          summary.cracked,                  summary.cracked ? "warn" : "ok"],
      ["min(z)",           (summary.min_z ?? 1).toFixed(3),  summary.cracked ? "warn" : "ok"],
      ["crack init step",  summary.crack_initiated_step ?? "-"],
      ["peak Fy",          summary.peak_reaction?.toExponential?.(3) ?? "-"],
      ["peak Fy disp",     summary.peak_reaction_disp?.toExponential?.(3) ?? "-"],
      ["final disp",       summary.final_disp?.toExponential?.(3) ?? "-"],
      ["steps",            summary.n_steps ?? "-"],
      ["diverged",         summary.diverged,                 summary.diverged ? "err" : "ok"],
    ];
    for (const [k, v, klass] of items) {
      const d = document.createElement("div");
      d.className = "metric" + (klass ? " " + klass : "");
      d.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
      metricsBox.appendChild(d);
    }
  }

  // ---- health score ------------------------------------------------ //
  function showHealth(h) {
    if (!h) { healthBox.classList.add("hidden"); return; }
    healthBox.classList.remove("hidden");
    const total = (h.total ?? 0).toFixed(1);
    healthScore.textContent = total;
    healthScore.className = "health-score " + (h.verdict || "");
    healthVerdict.textContent = (h.verdict || "?").toUpperCase();
    healthVerdict.className = "health-verdict " + (h.verdict || "");
    clear(healthFlags);
    (h.flags || []).forEach((f) => {
      const tag = document.createElement("span");
      tag.className = "health-flag";
      tag.textContent = f;
      healthFlags.appendChild(tag);
    });
    clear(healthBars);
    const components = ["integrity", "admissibility", "accuracy",
                         "mesh_independence", "efficiency"];
    for (const name of components) {
      const c = h[name];
      if (!c) continue;
      const pct = c.max_points ? (c.points / c.max_points) * 100 : 0;
      const tone = pct >= 85 ? "ok" : pct >= 50 ? "warn" : "err";
      const bar = document.createElement("div");
      bar.className = "health-bar";
      const label = c.name.replace(/_/g, " ");
      bar.innerHTML = (
        `<span class="name" title="${(c.notes || []).join('; ')}">${label}</span>` +
        `<div class="track"><div class="fill ${tone}" style="width:${pct.toFixed(0)}%"></div></div>` +
        `<span class="pts">${c.points.toFixed(1)} / ${c.max_points.toFixed(0)}</span>`
      );
      healthBars.appendChild(bar);
    }
  }

  // ---- assumptions ------------------------------------------------- //
  function appendAssumption(phase, text) {
    const ph = assumptionsLog.querySelector(".placeholder");
    if (ph) ph.remove();
    const entry = document.createElement("div");
    entry.className = "entry assumption";
    const phaseTag = document.createElement("span");
    phaseTag.className = "phase";
    phaseTag.textContent = phase || "";
    const msg = document.createElement("span");
    msg.className = "msg";
    msg.appendChild(phaseTag);
    msg.appendChild(document.createTextNode(text));
    entry.appendChild(msg);
    assumptionsLog.appendChild(entry);
    assumptionsLog.scrollTop = assumptionsLog.scrollHeight;
    bumpAssumptionCount();
  }
  function bumpAssumptionCount() {
    const n = assumptionsLog.querySelectorAll(".entry").length;
    assumptionsCount.textContent = n ? `(${n})` : "";
  }

  // ---- spec preview ------------------------------------------------ //
  function showSpec(spec, action, meshPlan) {
    if (!spec) return;
    specCard.classList.remove("hidden");
    const payload = { spec, action, mesh_plan: meshPlan };
    specPre.textContent = JSON.stringify(payload, null, 2);
  }

  // ---- images ------------------------------------------------------ //
  function showImage(name) {
    if (seenImages.has(name) || !sessionDirName) return;
    seenImages.add(name);
    imagesCard.classList.remove("hidden");
    const captions = {
      "final_mesh.png":   "Final mesh (undeformed)",
      "final_damage.png": "Deformed configuration · phase field z",
      "load_disp.png":    "Load–displacement curve",
    };
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = `/img/${encodeURIComponent(sessionDirName)}/${encodeURIComponent(name)}?t=${Date.now()}`;
    img.alt = name;
    img.addEventListener("click", () => openLightbox(img.src));
    const fc = document.createElement("figcaption");
    fc.textContent = captions[name] || name;
    fig.appendChild(img);
    fig.appendChild(fc);
    imagesBox.appendChild(fig);
  }
  function openLightbox(src) {
    const lb = document.createElement("div");
    lb.className = "lightbox";
    const img = document.createElement("img");
    img.src = src;
    lb.appendChild(img);
    lb.addEventListener("click", () => lb.remove());
    document.body.appendChild(lb);
  }

  // ---- downloads --------------------------------------------------- //
  function showDownloads(files) {
    if (!files || !files.length || !sessionDirName) return;
    downloadsCard.classList.remove("hidden");
    clear(downloadsBox);
    for (const f of files) {
      const a = document.createElement("a");
      a.className = "download " + (f.kind || "data");
      a.href = `/file/${encodeURIComponent(sessionDirName)}/${encodeURIComponent(f.name)}`;
      a.download = f.name;
      const icon = document.createElement("div");
      icon.className = "icon";
      icon.textContent = ({code: "PY", log: "LOG", data: "JSN", image: "IMG"})[f.kind] || "···";
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.innerHTML = `<div class="name">${f.name}</div><div class="size">${fmtBytes(f.size)}</div>`;
      a.appendChild(icon);
      a.appendChild(meta);
      downloadsBox.appendChild(a);
    }
  }

  // ---- run --------------------------------------------------------- //
  async function runSim() {
    const prompt = promptEl.value.trim();
    if (!prompt) {
      promptEl.focus();
      promptEl.style.borderColor = "var(--accent-error)";
      setTimeout(() => promptEl.style.borderColor = "", 900);
      return;
    }
    // Reset panels.
    clear(activityLog);
    clear(decisionsLog);
    clear(assumptionsLog);
    placeholder(assumptionsLog, "Defaults the agent applied appear here.");
    bumpAssumptionCount();
    clear(consoleLog);
    placeholder(verdictBox, "Running… verdict will appear here.");
    clear(metricsBox);
    healthBox.classList.add("hidden");
    clear(imagesBox);
    imagesCard.classList.add("hidden");
    downloadsCard.classList.add("hidden");
    specCard.classList.add("hidden");
    specPre.classList.add("hidden");
    specToggle.textContent = "show";
    chatCard.classList.add("hidden");
    clear(chatHistory);
    seenImages.clear();
    hadError = false;
    phasePill.textContent = "starting";
    updateCostMeter(null);

    runBtn.disabled = true;
    runBtn.textContent = "Running…";
    spinner.classList.remove("hidden");

    try {
      let resp;
      if (uploads.length > 0) {
        const fd = new FormData();
        fd.append("prompt", prompt);
        if (materialEl.value) fd.append("material", materialEl.value);
        if (geometryEl.value) fd.append("geometry", geometryEl.value);
        if (physicsEl.value)  fd.append("physics",  physicsEl.value);
        fd.append("fracture", fractureEl.checked ? "true" : "false");
        for (const f of uploads) fd.append("images", f, f.name);
        resp = await fetch("/run", { method: "POST", body: fd });
      } else {
        resp = await fetch("/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            prompt,
            material: materialEl.value,
            geometry: geometryEl.value,
            physics:  physicsEl.value,
            fracture: fractureEl.checked,
          }),
        });
      }
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "run failed");
      activeSession = data.session_id;
      sessionTag.textContent = activeSession;
      openStream(data.session_id);
    } catch (e) {
      appendEntry(activityLog, "error", "Failed to start: " + e.message);
      runBtn.disabled = false;
      runBtn.textContent = "Run simulation";
      spinner.classList.add("hidden");
      phasePill.textContent = "";
      showToast("Failed to start: " + e.message, "error");
    }
  }

  function inferPhase(message) {
    const m = (message || "").toLowerCase();
    if (m.includes("parsing")) return "receptionist";
    if (m.includes("architect")) return "architect";
    if (m.includes("handbook") || m.includes("material")) return "material";
    if (m.includes("strategist") || m.includes("variant")) return "strategist";
    if (m.includes("synthes")) return "synthesizer";
    if (m.includes("mesh") || m.includes("rescal")) return "mesh";
    if (m.includes("inspector")) return "inspector";
    if (m.includes("wsl") || m.includes("running") || m.includes("solver")) return "executor";
    if (m.includes("debug") || m.includes("uh oh")) return "debugger";
    if (m.includes("advisor") || m.includes("verdict") || m.includes("health")) return "advisor";
    if (m.includes("revis")) return "reviser";
    if (m.includes("rendering")) return "render";
    return null;
  }

  function openStream(sessionId) {
    const es = new EventSource(`/stream/${encodeURIComponent(sessionId)}`);
    es.onmessage = (ev) => {
      let d;
      try { d = JSON.parse(ev.data); } catch { return; }
      if (d.session_dir_name) sessionDirName = d.session_dir_name;
      if (d.session_id && sessionTag.textContent !== d.session_id) {
        sessionTag.textContent = d.session_id;
      }

      switch (d.category) {
        case "status": {
          appendEntry(activityLog, "status", d.message);
          const ph = inferPhase(d.message);
          if (ph) phasePill.textContent = ph;
          break;
        }
        case "decision":
          appendEntry(decisionsLog, "decision", d.message);
          break;
        case "console":
          appendConsole(d.message);
          break;
        case "result": {
          verdictBox.innerHTML = "";
          const p = document.createElement("div");
          p.textContent = d.message;
          verdictBox.appendChild(p);
          appendEntry(activityLog, "result", "Verdict ready.");
          break;
        }
        case "meta":
          try { showMetrics(JSON.parse(d.message)); } catch {}
          break;
        case "image":
          showImage(d.message);
          break;
        case "error":
          hadError = true;
          appendEntry(activityLog, "error", d.message);
          break;
        case "clarify":
          openClarify(d.questions || []);
          break;
        case "assumption":
          for (const a of (d.items || [])) {
            appendAssumption(d.phase || "", a);
          }
          if ((d.items || []).length) {
            appendEntry(activityLog, "decision",
              `${d.items.length} assumption(s) added in ${d.phase || "?"} phase.`);
          }
          break;
        case "telemetry":
          updateCostMeter(d.totals, d.iters, true);
          break;
        case "spec":
          showSpec(d.spec, d.action, d.mesh_plan);
          break;
        case "done":
          if (d.summary) showMetrics(d.summary);
          if (d.health)  showHealth(d.health);
          if (d.telemetry) updateCostMeter(d.telemetry, d.iters, false);
          if (d.files)  showDownloads(d.files);
          es.close();
          runBtn.disabled = false;
          runBtn.textContent = "Run simulation";
          spinner.classList.add("hidden");
          phasePill.textContent = "done";
          if (hadError) {
            appendEntry(activityLog, "error",
              "Run FAILED. The simulation itself may have finished; "
              + "check the console and the session folder "
              + "(agentic_simulations/" + (sessionDirName || "") + ").");
            verdictBox.innerHTML = "";
            const p = document.createElement("div");
            p.style.color = "var(--accent-error)";
            p.textContent = "Run failed before the advisor could report. "
              + "See the agent-activity panel above for the traceback.";
            verdictBox.appendChild(p);
            showToast("Run failed", "error");
          } else {
            chatCard.classList.remove("hidden");
            chatInput.focus();
            appendEntry(activityLog, "status",
              "Run complete. Ask me anything below.");
            showToast("Run complete", "ok");
          }
          break;
      }
    };
    es.onerror = () => {
      appendEntry(activityLog, "error", "Live stream disconnected.");
      es.close();
      runBtn.disabled = false;
      runBtn.textContent = "Run simulation";
      spinner.classList.add("hidden");
    };
  }

  // ---- chat -------------------------------------------------------- //
  async function sendQuestion(ev) {
    ev.preventDefault();
    if (!activeSession) return;
    const q = chatInput.value.trim();
    if (!q) return;
    chatInput.value = "";

    appendChat("you", q);
    const pending = appendChat("agent", "…");
    try {
      const r = await fetch(`/ask/${encodeURIComponent(activeSession)}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question: q}),
      });
      const data = await r.json();
      if (!r.ok) {
        pending.querySelector(".body").textContent = "error: " + (data.error || "?");
      } else {
        pending.querySelector(".body").textContent = data.answer;
      }
    } catch (e) {
      pending.querySelector(".body").textContent = "network error: " + e.message;
    }
  }

  function appendChat(role, text) {
    const row = document.createElement("div");
    row.className = `chat-msg ${role}`;
    const roleBadge = document.createElement("span");
    roleBadge.className = "role";
    roleBadge.textContent = role;
    const body = document.createElement("span");
    body.className = "body";
    body.textContent = text;
    row.appendChild(roleBadge);
    row.appendChild(body);
    chatHistory.appendChild(row);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return row;
  }

  // ---- clarification modal ---------------------------------------- //
  function openClarify(questions) {
    clear(clarifyQs);
    questions.forEach((q, i) => {
      const div = document.createElement("div");
      div.className = "clarify-q";
      const lbl = document.createElement("label");
      lbl.htmlFor = `clarify-input-${i}`;
      lbl.textContent = `Question ${i + 1}`;
      const qText = document.createElement("span");
      qText.className = "q-text";
      qText.textContent = q;
      const input = document.createElement("input");
      input.type = "text";
      input.id = `clarify-input-${i}`;
      input.placeholder = "Answer (or leave blank to use defaults)";
      input.dataset.idx = String(i);
      div.appendChild(lbl);
      div.appendChild(qText);
      div.appendChild(input);
      clarifyQs.appendChild(div);
    });
    clarifyModal.classList.remove("hidden");
    setTimeout(() => {
      const first = clarifyQs.querySelector("input[type=text]");
      if (first) first.focus();
    }, 50);
  }
  function closeClarify() {
    clarifyModal.classList.add("hidden");
  }
  async function submitClarify(answers) {
    if (!activeSession) { closeClarify(); return; }
    try {
      await fetch(`/clarify/${encodeURIComponent(activeSession)}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({answers}),
      });
    } catch (e) {
      showToast("Failed to send answers: " + e.message, "error");
    }
    closeClarify();
  }
  clarifyForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const answers = [...clarifyQs.querySelectorAll("input[type=text]")]
      .map((i) => i.value.trim());
    submitClarify(answers);
  });
  clarifySkip.addEventListener("click", () => {
    const n = clarifyQs.querySelectorAll("input").length;
    submitClarify(new Array(n).fill(""));
  });

  // ---- examples modal --------------------------------------------- //
  async function openExamples() {
    examplesModal.classList.remove("hidden");
    examplesList.innerHTML = '<div class="muted">Loading…</div>';
    try {
      const r = await fetch("/examples");
      const data = await r.json();
      const list = data.examples || [];
      if (!list.length) {
        examplesList.innerHTML = '<div class="muted">No examples available.</div>';
        return;
      }
      examplesList.innerHTML = "";
      for (const ex of list) {
        const card = document.createElement("div");
        card.className = "example-card";
        const tags = (ex.tags || []).map((t) => `<span class="tag">${t}</span>`).join("");
        card.innerHTML = (
          `<div class="label">${ex.label}</div>` +
          `<div class="tags">${tags}</div>` +
          `<div class="preview">${ex.prompt}</div>`
        );
        card.addEventListener("click", () => {
          promptEl.value = ex.prompt;
          examplesModal.classList.add("hidden");
          promptEl.focus();
          promptEl.scrollTop = 0;
        });
        examplesList.appendChild(card);
      }
    } catch (e) {
      examplesList.innerHTML = `<div class="muted">Failed to load: ${e.message}</div>`;
    }
  }

  // ---- history modal ---------------------------------------------- //
  async function openHistory() {
    historyModal.classList.remove("hidden");
    historyList.innerHTML = '<div class="muted">Loading…</div>';
    try {
      const r = await fetch("/sessions");
      const data = await r.json();
      const list = data.sessions || [];
      if (!list.length) {
        historyList.innerHTML = '<div class="muted">No past sessions yet.</div>';
        return;
      }
      historyList.innerHTML = "";
      for (const s of list) {
        const row = document.createElement("div");
        row.className = "history-item";
        const verdictTag = s.verdict
          ? `<span class="verdict-tag ${s.verdict}">${s.verdict}</span>`
          : '<span class="verdict-tag">—</span>';
        row.innerHTML = (
          `<div class="id" title="${s.id}">${s.id}</div>` +
          `<div class="meta">${s.material} · ${s.kind}</div>` +
          `<div class="meta">${s.constitutive}${s.fracture === false ? " · no-frac" : ""}</div>` +
          verdictTag
        );
        row.addEventListener("click", () => {
          // Open the session folder's state.json file as a download.
          window.open(`/file/${encodeURIComponent(s.id)}/state.json`, "_blank");
        });
        historyList.appendChild(row);
      }
    } catch (e) {
      historyList.innerHTML = `<div class="muted">Failed to load: ${e.message}</div>`;
    }
  }

  // ---- upload zone ------------------------------------------------- //
  function renderUploadList() {
    if (uploads.length === 0) {
      uploadEmpty.hidden = false;
      uploadList.hidden = true;
      uploadList.innerHTML = "";
      return;
    }
    uploadEmpty.hidden = true;
    uploadList.hidden = false;
    uploadList.innerHTML = "";
    uploads.forEach((f, i) => {
      const item = document.createElement("div");
      item.className = "upload-item";
      item.innerHTML = `<span class="name">${f.name}</span><span class="x" data-idx="${i}" title="remove">✕</span>`;
      uploadList.appendChild(item);
    });
  }
  function addUploads(fileList) {
    for (const f of fileList) {
      if (!f.type.startsWith("image/")) continue;
      uploads.push(f);
    }
    renderUploadList();
  }
  uploadZone.addEventListener("click", (e) => {
    if (e.target.classList.contains("x")) return;
    uploadInput.click();
  });
  uploadInput.addEventListener("change", () => {
    addUploads(uploadInput.files);
    uploadInput.value = "";
  });
  uploadList.addEventListener("click", (e) => {
    const x = e.target.closest(".x");
    if (!x) return;
    const idx = parseInt(x.dataset.idx, 10);
    uploads.splice(idx, 1);
    renderUploadList();
  });
  ["dragover", "dragenter"].forEach((ev) => {
    uploadZone.addEventListener(ev, (e) => {
      e.preventDefault();
      uploadZone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    uploadZone.addEventListener(ev, (e) => {
      e.preventDefault();
      uploadZone.classList.remove("dragover");
    });
  });
  uploadZone.addEventListener("drop", (e) => {
    if (e.dataTransfer && e.dataTransfer.files) {
      addUploads(e.dataTransfer.files);
    }
  });

  // ---- spec preview toggle ---------------------------------------- //
  specToggle.addEventListener("click", () => {
    const showing = !specPre.classList.contains("hidden");
    if (showing) {
      specPre.classList.add("hidden");
      specToggle.textContent = "show";
    } else {
      specPre.classList.remove("hidden");
      specToggle.textContent = "hide";
    }
  });

  // ---- activity filters ------------------------------------------- //
  filterToggles.addEventListener("change", (e) => {
    if (!(e.target instanceof HTMLInputElement)) return;
    const cat = e.target.dataset.filter;
    if (e.target.checked) activityFilters.add(cat);
    else activityFilters.delete(cat);
    activityLog.querySelectorAll(".entry").forEach((el) => {
      el.style.display = activityFilters.has(el.dataset.category) ? "" : "none";
    });
  });

  // ---- modal close handlers --------------------------------------- //
  function wireModalClose(modal, closeBtn) {
    closeBtn.addEventListener("click", () => modal.classList.add("hidden"));
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.classList.add("hidden");
    });
  }
  wireModalClose(examplesModal, examplesClose);
  wireModalClose(historyModal, historyClose);
  // (clarify modal is intentionally NOT close-on-backdrop — it would
  // leave the orchestrator parked.  The user must Submit or Skip.)

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      examplesModal.classList.add("hidden");
      historyModal.classList.add("hidden");
      // For clarify: Esc = "Skip all" — same as the explicit button.
      if (!clarifyModal.classList.contains("hidden")) {
        const n = clarifyQs.querySelectorAll("input").length;
        submitClarify(new Array(n).fill(""));
      }
    }
  });

  // ---- wire it up -------------------------------------------------- //
  runBtn.addEventListener("click", runSim);
  chatForm.addEventListener("submit", sendQuestion);
  consoleClear.addEventListener("click", () => {
    clear(consoleLog);
    placeholder(consoleLog, "Live WSL stdout streams here.");
  });
  promptEl.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runSim();
  });
  examplesBtn.addEventListener("click", openExamples);
  historyBtn.addEventListener("click", openHistory);

  // initial state
  updateCostMeter(null);
  renderUploadList();
})();
