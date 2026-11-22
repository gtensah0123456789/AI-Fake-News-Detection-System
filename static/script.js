/**
 * script.js
 * Frontend interactivity for News Detection and Analytics Dashboard.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ---------------- Detector Page Logic ----------------
  const newsForm = document.getElementById("newsForm");
  const newsInput = document.getElementById("newsInput");
  const charCount = document.getElementById("charCount");
  const clearBtn = document.getElementById("clearBtn");
  const analyseBtn = document.getElementById("analyseBtn");
  const sampleRealBtn = document.getElementById("sampleRealBtn");
  const sampleFakeBtn = document.getElementById("sampleFakeBtn");
  const errorAlert = document.getElementById("errorAlert");
  const errorMessage = document.getElementById("errorMessage");
  const loadingBox = document.getElementById("loadingBox");
  const resultSection = document.getElementById("resultSection");
  const resultCard = document.getElementById("resultCard");
  const predictionLabel = document.getElementById("predictionLabel");
  const confidenceBadge = document.getElementById("confidenceBadge");
  const confidenceBar = document.getElementById("confidenceBar");
  const predictionExplanation = document.getElementById("predictionExplanation");
  const analyseAnotherBtn = document.getElementById("analyseAnotherBtn");

  const SAMPLE_REAL =
    "The Reserve Bank of India kept the benchmark repo rate unchanged at 6.5 percent during its bi-monthly monetary policy committee meeting, citing steady domestic economic growth and ongoing inflation monitoring.";
  const SAMPLE_FAKE =
    "BREAKING SHOCKING: Secret government memo proves 5G towers are secretly controlling human dreams and draining bank accounts overnight! Share this urgent warning before deleted!";

  function showError(msg) {
    if (!errorAlert || !errorMessage) return;
    errorMessage.textContent = msg;
    errorAlert.classList.remove("d-none");
    errorAlert.classList.add("d-flex");
  }

  function hideError() {
    if (!errorAlert) return;
    errorAlert.classList.add("d-none");
    errorAlert.classList.remove("d-flex");
  }

  function updateCharCounter() {
    if (newsInput && charCount) {
      charCount.textContent = `${newsInput.value.length.toLocaleString()} characters`;
    }
  }

  if (newsInput) {
    newsInput.addEventListener("input", () => {
      updateCharCounter();
      hideError();
    });
  }

  if (sampleRealBtn) {
    sampleRealBtn.addEventListener("click", () => {
      newsInput.value = SAMPLE_REAL;
      updateCharCounter();
      hideError();
    });
  }

  if (sampleFakeBtn) {
    sampleFakeBtn.addEventListener("click", () => {
      newsInput.value = SAMPLE_FAKE;
      updateCharCounter();
      hideError();
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      newsInput.value = "";
      updateCharCounter();
      hideError();
      if (resultSection) resultSection.classList.add("d-none");
      newsInput.focus();
    });
  }

  if (analyseAnotherBtn) {
    analyseAnotherBtn.addEventListener("click", () => {
      newsInput.value = "";
      updateCharCounter();
      hideError();
      resultSection.classList.add("d-none");
      newsInput.focus();
    });
  }

  if (newsForm) {
    newsForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      hideError();

      const text = newsInput.value.trim();
      if (!text) {
        showError("Please paste a news headline or article before clicking Analyse News.");
        return;
      }
      if (text.length < 15) {
        showError("Please enter at least 15 characters so the model has enough context.");
        return;
      }

      // Show loading animation
      resultSection.classList.add("d-none");
      loadingBox.classList.remove("d-none");
      analyseBtn.disabled = true;

      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Failed to analyze the news text.");
        }

        renderPredictionResult(data);
      } catch (err) {
        showError(err.message || "Network error while contacting the prediction server.");
      } finally {
        loadingBox.classList.add("d-none");
        analyseBtn.disabled = false;
      }
    });
  }

  function renderPredictionResult(data) {
    const isReal = data.raw_label === "REAL";
    resultCard.classList.remove("result-real", "result-fake");
    resultCard.classList.add(isReal ? "result-real" : "result-fake");

    predictionLabel.textContent = data.prediction;
    predictionLabel.className = `display-6 fw-bold mb-0 mt-1 ${
      isReal ? "text-success" : "text-danger"
    }`;

    confidenceBadge.textContent = `${data.confidence}%`;
    confidenceBadge.className = `badge fs-5 px-3 py-2 mt-1 ${
      isReal ? "bg-success" : "bg-danger"
    }`;

    confidenceBar.style.width = `${data.confidence}%`;
    confidenceBar.className = `progress-bar progress-bar-striped progress-bar-animated ${
      isReal ? "bg-success" : "bg-danger"
    }`;

    predictionExplanation.textContent = data.explanation;
    resultSection.classList.remove("d-none");
    resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // ---------------- Dashboard Page Logic ----------------
  const statTotal = document.getElementById("statTotal");
  if (statTotal) {
    let distributionChart = null;
    let confidenceChart = null;

    async function loadDashboardData() {
      try {
        const [statsRes, historyRes] = await Promise.all([
          fetch("/api/stats"),
          fetch("/api/history?limit=25"),
        ]);

        const stats = await statsRes.json();
        const historyPayload = await historyRes.json();

        // Update KPI numbers
        document.getElementById("statTotal").textContent = stats.total;
        document.getElementById("statReal").textContent = stats.likely_real;
        document.getElementById("statFake").textContent = stats.likely_fake;
        document.getElementById("statAvgConf").textContent = `${stats.avg_confidence}%`;

        renderCharts(stats);
        renderHistoryTable(historyPayload.history || []);
      } catch (err) {
        console.error("Dashboard load error:", err);
      }
    }

    function renderCharts(stats) {
      const distCtx = document.getElementById("distributionChart");
      const confCtx = document.getElementById("confidenceChart");

      if (distributionChart) distributionChart.destroy();
      if (confidenceChart) confidenceChart.destroy();

      distributionChart = new Chart(distCtx, {
        type: "doughnut",
        data: {
          labels: ["Likely Real", "Likely Fake"],
          datasets: [
            {
              data: [stats.likely_real, stats.likely_fake],
              backgroundColor: ["#22c55e", "#ef4444"],
              borderWidth: 2,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: "bottom" },
          },
        },
      });

      const trend = stats.trend || [];
      const labels = trend.map((item) => `#${item.id}`);
      const values = trend.map((item) => item.confidence);
      const colors = trend.map((item) =>
        item.prediction === "Likely Real" ? "#22c55e" : "#ef4444"
      );

      confidenceChart = new Chart(confCtx, {
        type: "bar",
        data: {
          labels,
          datasets: [
            {
              label: "Confidence (%)",
              data: values,
              backgroundColor: colors,
              borderRadius: 6,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
            },
          },
          plugins: {
            legend: { display: false },
          },
        },
      });
    }

    function renderHistoryTable(records) {
      const tbody = document.getElementById("historyTableBody");
      const countBadge = document.getElementById("historyCountBadge");
      countBadge.textContent = `${records.length} Record${records.length === 1 ? "" : "s"}`;

      if (!records.length) {
        tbody.innerHTML = `
          <tr>
            <td colspan="5" class="text-center text-muted py-4">
              No predictions recorded yet. Go to <strong>Detect News</strong> to analyze an article.
            </td>
          </tr>`;
        return;
      }

      tbody.innerHTML = records
        .map((row) => {
          const isReal = row.prediction === "Likely Real";
          const badgeClass = isReal ? "bg-success" : "bg-danger";
          const safeText = escapeHtml(row.article_text);
          return `
            <tr>
              <td class="fw-semibold text-muted">#${row.id}</td>
              <td><div class="article-snippet" title="${safeText}">${safeText}</div></td>
              <td><span class="badge ${badgeClass}">${row.prediction}</span></td>
              <td class="fw-semibold">${row.confidence}%</td>
              <td class="small text-muted">${row.created_at}</td>
            </tr>`;
        })
        .join("");
    }

    function escapeHtml(str) {
      return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    const refreshBtn = document.getElementById("refreshDashboardBtn");
    if (refreshBtn) {
      refreshBtn.addEventListener("click", loadDashboardData);
    }

    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    if (clearHistoryBtn) {
      clearHistoryBtn.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to delete all prediction history?")) return;
        try {
          const res = await fetch("/api/history/clear", { method: "DELETE" });
          if (res.ok) {
            loadDashboardData();
          }
        } catch (err) {
          alert("Failed to clear history.");
        }
      });
    }

    loadDashboardData();
  }
});
