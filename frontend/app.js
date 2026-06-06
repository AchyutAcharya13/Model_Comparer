const statusEl = document.getElementById("status");
const optimizerEl = document.getElementById("optimizer");
const epochsEl = document.getElementById("epochs");
const subsetSizeEl = document.getElementById("subsetSize");
const testSubsetSizeEl = document.getElementById("testSubsetSize");
const batchSizeEl = document.getElementById("batchSize");
const trialsEl = document.getElementById("trials");
const seedEl = document.getElementById("seed");
const deterministicEl = document.getElementById("deterministic");
const fullDatasetEl = document.getElementById("fullDataset");
const useAugmentationEl = document.getElementById("useAugmentation");
const paperProtocolEl = document.getElementById("paperProtocol");
const trainBtn = document.getElementById("trainBtn");
const compareBtn = document.getElementById("compareBtn");
const healthBtn = document.getElementById("healthBtn");
const metricsPanel = document.getElementById("metricsPanel");
const historyPanel = document.getElementById("historyPanel");
const historyBody = document.getElementById("historyBody");
const artifactListEl = document.getElementById("artifactList");
const summaryEls = {
  trials: document.getElementById("sTrials"),
  trainAcc: document.getElementById("sTrainAcc"),
  valAcc: document.getElementById("sValAcc"),
  valLoss: document.getElementById("sValLoss"),
};
const comparePanel = document.getElementById("comparePanel");
const compareBody = document.getElementById("compareBody");
const bestOptimizerEl = document.getElementById("bestOptimizer");

const metricEls = {
  trainLoss: document.getElementById("mTrainLoss"),
  trainAcc: document.getElementById("mTrainAcc"),
  valLoss: document.getElementById("mValLoss"),
  valAcc: document.getElementById("mValAcc"),
};

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "var(--danger)" : "var(--muted)";
}

function getTrainConfig() {
  return {
    epochs: Number(epochsEl.value),
    subset_size: Number(subsetSizeEl.value),
    test_subset_size: Number(testSubsetSizeEl.value),
    batch_size: Number(batchSizeEl.value),
    trials: Number(trialsEl.value),
    seed: Number(seedEl.value),
    deterministic: deterministicEl.checked,
    full_dataset: fullDatasetEl.checked,
    use_augmentation: useAugmentationEl.checked,
    paper_protocol: paperProtocolEl.checked,
  };
}

function formatMeanStd(meanValue, stdValue, suffix = "") {
  if (typeof meanValue !== "number") {
    return "-";
  }
  const mean = meanValue.toFixed(2);
  const std = typeof stdValue === "number" ? stdValue.toFixed(2) : "0.00";
  return `${mean} ± ${std}${suffix}`;
}

function renderArtifacts(data) {
  artifactListEl.innerHTML = "";
  const items = [];
  if (data.artifact_path) {
    items.push("Aggregate: " + data.artifact_path);
  }
  if (Array.isArray(data.trial_artifact_paths)) {
    for (const trialPath of data.trial_artifact_paths) {
      items.push("Trial: " + trialPath);
    }
  }
  if (items.length === 0) {
    items.push("No artifacts returned.");
  }
  for (const line of items) {
    const li = document.createElement("li");
    li.textContent = line;
    artifactListEl.appendChild(li);
  }
}

function renderSummary(data) {
  const summary = data.summary || {};
  summaryEls.trials.textContent = String(summary.num_trials ?? data.trials ?? 1);
  summaryEls.trainAcc.textContent = formatMeanStd(
    summary.final_train_acc_mean,
    summary.final_train_acc_std,
    "%"
  );
  summaryEls.valAcc.textContent = formatMeanStd(
    summary.final_val_acc_mean,
    summary.final_val_acc_std,
    "%"
  );
  summaryEls.valLoss.textContent = formatMeanStd(
    summary.final_val_loss_mean,
    summary.final_val_loss_std
  );
  renderArtifacts(data);
}

async function loadOptimizers() {
  try {
    const res = await fetch("/optimizers");
    if (!res.ok) {
      throw new Error("Failed to fetch optimizers");
    }
    const data = await res.json();
    optimizerEl.innerHTML = "";
    for (const name of data.optimizers) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name.toUpperCase();
      optimizerEl.appendChild(option);
    }
    setStatus("Ready. Choose optimizer and start training.");
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.status === "ok") {
      setStatus("API health check passed.");
    } else {
      setStatus("API health check returned unexpected status.", true);
    }
  } catch (err) {
    setStatus("Health check failed: " + err.message, true);
  }
}

function renderResults(data) {
  metricsPanel.hidden = false;
  historyPanel.hidden = false;

  metricEls.trainLoss.textContent = data.final_train_loss.toFixed(4);
  metricEls.trainAcc.textContent = data.final_train_acc.toFixed(2) + "%";
  metricEls.valLoss.textContent = data.final_val_loss.toFixed(4);
  metricEls.valAcc.textContent = data.final_val_acc.toFixed(2) + "%";

  renderSummary(data);

  historyBody.innerHTML = "";
  for (const row of data.history) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.epoch}</td>
      <td>${row.train_loss.toFixed(4)}</td>
      <td>${row.train_acc.toFixed(2)}</td>
      <td>${row.val_loss.toFixed(4)}</td>
      <td>${row.val_acc.toFixed(2)}</td>
    `;
    historyBody.appendChild(tr);
  }
}

function renderComparison(data) {
  comparePanel.hidden = false;
  bestOptimizerEl.textContent = "Best optimizer: " + data.best_optimizer.toUpperCase();
  compareBody.innerHTML = "";

  for (const row of data.results) {
    const summary = row.summary || {};
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.optimizer.toUpperCase()}</td>
      <td>${summary.num_trials ?? row.trials ?? 1}</td>
      <td>${formatMeanStd(summary.final_train_acc_mean, summary.final_train_acc_std, "%")}</td>
      <td>${formatMeanStd(summary.final_val_acc_mean, summary.final_val_acc_std, "%")}</td>
      <td>${formatMeanStd(summary.final_train_loss_mean, summary.final_train_loss_std)}</td>
      <td>${formatMeanStd(summary.final_val_loss_mean, summary.final_val_loss_std)}</td>
    `;
    compareBody.appendChild(tr);
  }
}

async function train() {
  trainBtn.disabled = true;
  compareBtn.disabled = true;
  setStatus("Training started. Reproduction-style runs can take significantly longer...");

  try {
    const config = getTrainConfig();
    const res = await fetch("/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        optimizer: optimizerEl.value,
        epochs: config.epochs,
        subset_size: config.subset_size,
        test_subset_size: config.test_subset_size,
        batch_size: config.batch_size,
        trials: config.trials,
        seed: config.seed,
        deterministic: config.deterministic,
        full_dataset: config.full_dataset,
        use_augmentation: config.use_augmentation,
        paper_protocol: config.paper_protocol,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Training failed");
    }

    renderResults(data);
    setStatus("Training complete using " + data.optimizer.toUpperCase() + ".");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    trainBtn.disabled = false;
    compareBtn.disabled = false;
  }
}

async function compareOptimizers() {
  trainBtn.disabled = true;
  compareBtn.disabled = true;
  setStatus("Running multi-trial optimizer comparison (Adam/SGD/AdaBS). This will take longer...");

  try {
    const config = getTrainConfig();
    const res = await fetch("/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        epochs: config.epochs,
        subset_size: config.subset_size,
        test_subset_size: config.test_subset_size,
        batch_size: config.batch_size,
        trials: config.trials,
        seed: config.seed,
        deterministic: config.deterministic,
        full_dataset: config.full_dataset,
        use_augmentation: config.use_augmentation,
        paper_protocol: config.paper_protocol,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Comparison failed");
    }

    renderComparison(data);
    setStatus("Comparison complete. Best optimizer: " + data.best_optimizer.toUpperCase() + ".");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    trainBtn.disabled = false;
    compareBtn.disabled = false;
  }
}

trainBtn.addEventListener("click", train);
compareBtn.addEventListener("click", compareOptimizers);
healthBtn.addEventListener("click", checkHealth);

paperProtocolEl.addEventListener("change", () => {
  if (paperProtocolEl.checked) {
    epochsEl.max = "500";
    if (Number(epochsEl.value) < 200) {
      epochsEl.value = "200";
    }
    fullDatasetEl.checked = true;
  } else {
    epochsEl.max = "50";
    if (Number(epochsEl.value) > 50) {
      epochsEl.value = "50";
    }
  }
});

loadOptimizers();
