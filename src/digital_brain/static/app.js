const chatLog = document.querySelector("#chatLog");
const chatForm = document.querySelector("#chatForm");
const chatInput = document.querySelector("#chatInput");
const loopStatus = document.querySelector("#loopStatus");
const hormonesEl = document.querySelector("#hormones");
const goalsEl = document.querySelector("#goals");
const workingMemoryEl = document.querySelector("#workingMemory");
const memoryTable = document.querySelector("#memoryTable");
const memoryItems = document.querySelector("#memoryItems");
const systemOutput = document.querySelector("#systemOutput");
const actionOutput = document.querySelector("#actionOutput");

function addMessage(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = text;
  chatLog.appendChild(node);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

function renderState(state) {
  loopStatus.textContent = state.running ? "autonomous" : "manual";
  hormonesEl.innerHTML = "";
  Object.entries(state.hormones).forEach(([name, value]) => {
    const wrapper = document.createElement("div");
    wrapper.innerHTML = `
      <div class="metric-label"><span>${name}</span><span>${Math.round(value * 100)}%</span></div>
      <div class="bar"><span style="width:${Math.round(value * 100)}%"></span></div>
    `;
    hormonesEl.appendChild(wrapper);
  });

  goalsEl.innerHTML = "";
  state.goals.forEach((goal) => {
    const item = document.createElement("li");
    item.textContent = goal;
    goalsEl.appendChild(item);
  });

  workingMemoryEl.innerHTML = "";
  if (state.working_memory.length === 0) {
    workingMemoryEl.textContent = "No active chunks yet.";
  } else {
    state.working_memory.forEach((item) => {
      const node = document.createElement("div");
      node.className = "stack-item";
      node.textContent = `${item.kind}: ${item.content} (${item.relevance.toFixed(2)})`;
      workingMemoryEl.appendChild(node);
    });
  }
}

async function refreshState() {
  const data = await api("/api/state");
  renderState(data);
}

async function refreshMemory() {
  const data = await api(`/api/memory/${memoryTable.value}?limit=12`);
  memoryItems.innerHTML = "";
  data.items.forEach((item) => {
    const node = document.createElement("div");
    node.className = "memory-item";
    node.textContent = JSON.stringify(item, null, 2);
    memoryItems.appendChild(node);
  });
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  chatInput.value = "";
  addMessage("user", message);
  const data = await api("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message }),
  });
  addMessage("agent", data.reply);
  renderState(data.state);
  refreshMemory();
});

document.querySelector("#startLoop").addEventListener("click", async () => {
  renderState(await api("/api/loop/start", { method: "POST" }));
});

document.querySelector("#stopLoop").addEventListener("click", async () => {
  renderState(await api("/api/loop/stop", { method: "POST" }));
});

document.querySelector("#dream").addEventListener("click", async () => {
  const data = await api("/api/dream", { method: "POST" });
  renderState(data.state);
  refreshMemory();
});

document.querySelector("#reflect").addEventListener("click", async () => {
  const data = await api("/api/reflect", { method: "POST" });
  renderState(data.state);
  refreshMemory();
});

document.querySelector("#observeSystem").addEventListener("click", async () => {
  const data = await api("/api/system");
  systemOutput.textContent = JSON.stringify(data, null, 2);
  refreshMemory();
});

memoryTable.addEventListener("change", refreshMemory);

document.querySelector("#runAction").addEventListener("click", async () => {
  let parameters;
  try {
    parameters = JSON.parse(document.querySelector("#actionParams").value);
  } catch (error) {
    actionOutput.textContent = `Invalid JSON: ${error.message}`;
    return;
  }
  const data = await api("/api/actions", {
    method: "POST",
    body: JSON.stringify({
      action_type: document.querySelector("#actionType").value,
      parameters,
    }),
  });
  actionOutput.textContent = JSON.stringify(data, null, 2);
  refreshMemory();
});

addMessage("agent", "Digital Brain online. I can think, remember, plan, observe the OS, and perform safe approved actions.");
refreshState();
refreshMemory();
