(function () {
  // ลงทะเบียน Chart.js Annotation Plugin
  if (
    typeof Chart !== "undefined" &&
    typeof ChartAnnotation !== "undefined"
  ) {
    Chart.register(ChartAnnotation);
  } else if (typeof Chart !== "undefined" && window.ChartAnnotation) {
    Chart.register(window.ChartAnnotation);
  }

  let currentPredictionData = [];
  let globalMaxPred = 1;
  let currentHistoricalData = [];
  let allYearsList = [];
  let firstPredYearValue = null;
  let chartInstance = null;
  let totalChartInstance = null;

  const useNewDataCheck = document.getElementById("useNewDataCheck");
  const fileInput = document.getElementById("fileInput");
  const uploadBtn = document.getElementById("uploadBtn");
  const loadingDiv = document.getElementById("loading");
  const modelName =
    document.getElementById("modelSelect").value === "gnn"
      ? "Graph Wavenet GNN"
      : "XGBoost";
  loadingDiv.querySelector("p").textContent =
    `กำลังประมวลผลด้วย ${modelName}...`;
  const evalSection = document.getElementById("evalSection");
  const predSection = document.getElementById("predSection");
  const pageErrorDiv = document.getElementById("pageError");
  const provinceCheckboxContainer = document.getElementById(
    "provinceCheckboxList",
  );
  const provinceSearchInput = document.getElementById("provinceSearch");
  const selectAllBtn = document.getElementById("selectAllProvincesBtn");
  const clearAllBtn = document.getElementById("clearAllProvincesBtn");
  const selectTop5CheckboxBtn = document.getElementById(
    "selectTop5CheckboxBtn",
  );
  const nYearsInput = document.getElementById("nYearsInput");
  const startYearInput = document.getElementById("startYearInput");
  const endYearInput = document.getElementById("endYearInput");

  let uploadInProgress = false;
  let allProvinceNames = [];
  let edgarData = null; // { year → CO2_tonnes }
  let edgarVisible = false;
  let gcaData = null; // { year → CO2_tonnes }
  let gcaVisible = false;

  function renderProvinceCheckboxes(provinces) {
    allProvinceNames = provinces.sort((a, b) => a.localeCompare(b, "th"));
    provinceCheckboxContainer.innerHTML = "";
    allProvinceNames.forEach((prov) => {
      const div = document.createElement("div");
      div.className = "form-check province-check-item";
      div.innerHTML = `<input class="form-check-input" type="checkbox" value="${prov}" id="chk_${prov.replace(/\s+/g, "_")}"><label class="form-check-label" for="chk_${prov.replace(/\s+/g, "_")}">${prov}</label>`;
      provinceCheckboxContainer.appendChild(div);
    });
    document
      .querySelectorAll(".province-check-item input[type=checkbox]")
      .forEach((cb) =>
        cb.addEventListener("change", updateChartFromCheckboxes),
      );
  }

  function filterCheckboxes(searchTerm) {
    const items = provinceCheckboxContainer.querySelectorAll(
      ".province-check-item",
    );
    const lowerTerm = searchTerm.toLowerCase();
    items.forEach((item) => {
      const label = item.querySelector("label").textContent.toLowerCase();
      item.style.display = label.includes(lowerTerm) ? "flex" : "none";
    });
  }

  function getSelectedProvincesFromCheckboxes() {
    return Array.from(
      provinceCheckboxContainer.querySelectorAll(
        "input[type=checkbox]:checked",
      ),
    ).map((cb) => cb.value);
  }

  function setCheckboxesSelection(provinceArray) {
    provinceCheckboxContainer
      .querySelectorAll("input[type=checkbox]")
      .forEach((cb) => (cb.checked = provinceArray.includes(cb.value)));
  }

  function updateChartFromCheckboxes() {
    const selected = getSelectedProvincesFromCheckboxes();
    if (selected.length === 0) {
      setCheckboxesSelection(allProvinceNames);
      redrawChart(allProvinceNames);
    } else redrawChart(selected);
  }

  function persistLog(msg, type = "info") {
    const logs = JSON.parse(localStorage.getItem("app_logs") || "[]");
    logs.push({ time: new Date().toISOString(), type, message: msg });
    if (logs.length > 50) logs.shift();
    localStorage.setItem("app_logs", JSON.stringify(logs));
    console.log(`[${type}] ${msg}`);
  }

  function getTop5Provinces(predictionData) {
    const byProvince = {};
    predictionData.forEach(
      (d) =>
        (byProvince[d.province] =
          (byProvince[d.province] || 0) + d.preds),
    );
    return Object.entries(byProvince)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map((e) => e[0]);
  }

  function redrawChart(selectedProvinces) {
    const canvas = document.getElementById("top5Chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!selectedProvinces || selectedProvinces.length === 0) {
      if (chartInstance) chartInstance.destroy();
      chartInstance = null;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.font = "16px Sarabun, sans-serif";
      ctx.fillStyle = "#999";
      ctx.textAlign = "center";
      ctx.fillText(
        "กรุณาเลือกอย่างน้อย 1 จังหวัดเพื่อแสดงกราฟ",
        canvas.width / 2,
        canvas.height / 2,
      );
      return;
    }
    if (!currentPredictionData.length || !allYearsList.length) return;

    const colors = [
      "#2E7D32",
      "#1565C0",
      "#E65100",
      "#6A1B9A",
      "#00695C",
      "#B71C1C",
      "#00838F",
      "#F57F17",
      "#4A148C",
      "#827717",
    ];
    const datasets = selectedProvinces.map((prov, i) => {
      const data = allYearsList.map((yr) => {
        const h = currentHistoricalData.find(
          (d) => d.province === prov && d.year === yr,
        );
        if (h) return h.actual / 1_000_000;
        const p = currentPredictionData.find(
          (d) => d.province === prov && d.year === yr,
        );
        return p ? p.preds / 1_000_000 : null;
      });
      return {
        label: prov,
        data,
        borderColor: colors[i % colors.length],
        backgroundColor: "transparent",
        borderWidth: 3,
        pointRadius: (ctx) =>
          allYearsList[ctx.dataIndex] >= firstPredYearValue ? 5 : 2,
        pointBackgroundColor: (ctx) =>
          allYearsList[ctx.dataIndex] >= firstPredYearValue
            ? colors[i % colors.length]
            : "#fff",
        pointBorderColor: colors[i % colors.length],
        pointBorderWidth: 2,
        tension: 0.2,
        segment: {
          borderDash: (ctx) => {
            const year = allYearsList[ctx.p0DataIndex];
            return year >= firstPredYearValue ? [5, 3] : undefined;
          },
        },
      };
    });
    if (chartInstance) chartInstance.destroy();
    chartInstance = new Chart(ctx, {
      type: "line",
      data: { labels: allYearsList, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        scales: {
          y: {
            beginAtZero: false,
            title: { display: true, text: "ล้านตัน CO₂" },
            ticks: {
              callback: (v) =>
                v.toLocaleString(undefined, { maximumFractionDigits: 2 }),
            },
          },
          x: { title: { display: true, text: "ปี" } },
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: (c) => {
                const yr = allYearsList[c.dataIndex];
                const t =
                  yr >= firstPredYearValue ? "คาดการณ์" : "ประวัติ";
                return `${c.dataset.label} (${t}): ${c.raw.toLocaleString(undefined, { maximumFractionDigits: 2 })} ล้านตัน`;
              },
            },
          },
        },
      },
    });
  }

  function drawTotalChart() {
    const canvas = document.getElementById("totalCountryChart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!currentHistoricalData.length && !currentPredictionData.length) {
      if (totalChartInstance) totalChartInstance.destroy();
      totalChartInstance = null;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.font = "16px Sarabun, sans-serif";
      ctx.fillStyle = "#999";
      ctx.textAlign = "center";
      ctx.fillText(
        "ไม่มีข้อมูลสำหรับแสดงกราฟรวม",
        canvas.width / 2,
        canvas.height / 2,
      );
      return;
    }

    const histTotals = {};
    currentHistoricalData.forEach(
      (d) =>
        (histTotals[d.year] =
          (histTotals[d.year] || 0) + (Number(d.actual) || 0)),
    );
    const predTotals = {};
    currentPredictionData.forEach(
      (d) =>
        (predTotals[d.year] =
          (predTotals[d.year] || 0) + (Number(d.preds) || 0)),
    );

    const allYears = [
      ...new Set([
        ...Object.keys(histTotals).map(Number),
        ...Object.keys(predTotals).map(Number),
      ]),
    ].sort((a, b) => a - b);
    const predYears = Object.keys(predTotals).map(Number);
    const firstPredYear = predYears.length
      ? Math.min(...predYears)
      : null;

    let histData = allYears.map((yr) =>
      histTotals[yr] ? histTotals[yr] / 1_000_000 : null,
    );
    let predData = allYears.map((yr) =>
      predTotals[yr] ? predTotals[yr] / 1_000_000 : null,
    );

    if (
      Object.keys(histTotals).length &&
      Object.keys(predTotals).length
    ) {
      const lastHistYear = Math.max(
        ...Object.keys(histTotals).map(Number),
      );
      const firstPredYearVal = Math.min(
        ...Object.keys(predTotals).map(Number),
      );
      if (firstPredYearVal === lastHistYear + 1) {
        const boundaryIndex = allYears.indexOf(lastHistYear);
        if (boundaryIndex !== -1)
          predData[boundaryIndex] = histTotals[lastHistYear] / 1_000_000;
      }
    }

    const datasets = [
      {
        label: "ข้อมูลประวัติ",
        data: histData,
        borderColor: "#0d6efd",
        backgroundColor: "transparent",
        borderWidth: 3,
        pointRadius: 3,
        pointBackgroundColor: "#0d6efd",
        tension: 0.2,
        spanGaps: true,
      },
      {
        label: "ค่าคาดการณ์",
        data: predData,
        borderColor: "#dc3545",
        backgroundColor: "transparent",
        borderWidth: 3,
        borderDash: [5, 3],
        pointRadius: 4,
        pointBackgroundColor: "#dc3545",
        tension: 0.2,
        spanGaps: true,
      },
    ];

    if (edgarVisible && edgarData) {
      const edgarLine = allYears.map((yr) =>
        edgarData[yr] != null ? edgarData[yr] / 1_000_000 : null,
      );
      datasets.push({
        label: "EDGAR (ค่าจริง)",
        data: edgarLine,
        borderColor: "#fd7e14",
        backgroundColor: "transparent",
        borderWidth: 2,
        borderDash: [],
        pointRadius: 3,
        pointBackgroundColor: "#fd7e14",
        tension: 0.2,
        spanGaps: true,
      });
    }

    if (gcaVisible && gcaData) {
      const gcaLine = allYears.map((yr) =>
        gcaData[yr] != null ? gcaData[yr] / 1_000_000 : null,
      );
      datasets.push({
        label: "GCA CO2",
        data: gcaLine,
        borderColor: "#28a745",
        backgroundColor: "transparent",
        borderWidth: 2,
        borderDash: [],
        pointRadius: 3,
        pointBackgroundColor: "#28a745",
        tension: 0.2,
        spanGaps: true,
      });
    }

    const annotation = firstPredYear
      ? {
          annotations: {
            predLine: {
              type: "line",
              xMin: firstPredYear,
              xMax: firstPredYear,
              borderColor: "#6c757d",
              borderWidth: 2,
              borderDash: [3, 3],
              label: {
                content: "เริ่มคาดการณ์",
                display: true,
                position: "top",
                backgroundColor: "rgba(0,0,0,0.7)",
                color: "white",
                font: { size: 11 },
              },
            },
          },
        }
      : {};

    if (totalChartInstance) totalChartInstance.destroy();
    totalChartInstance = new Chart(ctx, {
      type: "line",
      data: { labels: allYears, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        interaction: { intersect: false, mode: "index" },
        scales: {
          y: {
            beginAtZero: false,
            title: { display: true, text: "ล้านตัน CO₂" },
            ticks: {
              callback: (v) =>
                v.toLocaleString(undefined, { maximumFractionDigits: 2 }),
            },
          },
          x: { title: { display: true, text: "ปี" } },
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const val = ctx.raw;
                const year = allYears[ctx.dataIndex];
                const type =
                  firstPredYear !== null && year >= firstPredYear
                    ? "คาดการณ์"
                    : "ประวัติ";
                return `${ctx.dataset.label}: ${val.toLocaleString(undefined, { maximumFractionDigits: 2 })} ล้านตัน`;
              },
            },
          },
          annotation: annotation,
        },
      },
    });
  }

  selectAllBtn.addEventListener("click", () => {
    setCheckboxesSelection(allProvinceNames);
    redrawChart(allProvinceNames);
  });
  clearAllBtn.addEventListener("click", () => {
    setCheckboxesSelection([]);
    redrawChart([]);
  });
  selectTop5CheckboxBtn.addEventListener("click", () => {
    if (!currentPredictionData.length) return;
    const top5 = getTop5Provinces(currentPredictionData);
    setCheckboxesSelection(top5);
    redrawChart(top5);
  });

  async function loadEdgarData() {
    if (edgarData) return edgarData; // cache
    try {
      const res = await fetch("/static/edgar.txt");
      if (!res.ok)
        throw new Error(
          "โหลด edgar.txt ไม่ได้ (HTTP " + res.status + ")",
        );
      const text = await res.text();
      const lines = text.trim().split("\n");
      const result = {};
      for (const line of lines) {
        if (!line.trim() || line.startsWith("year")) continue;
        const parts = line.trim().split(/\s+/);
        const year = parseInt(parts[0], 10);
        const val = parseFloat(parts.slice(1).join("").replace(/,/g, ""));
        if (!isNaN(year) && !isNaN(val)) {
          result[year] = val;
        }
      }
      edgarData = result;
      console.log("EDGAR DATA:", edgarData);
      return edgarData;
    } catch (e) {
      alert(
        "ไม่สามารถโหลด edgar.txt: " +
          e.message +
          "\nกรุณาวางไฟล์ edgar.txt ไว้ในโฟลเดอร์ static/ เดียวกับ index.html",
      );
      return null;
    }
  }

  async function loadGCAData() {
    if (gcaData) return gcaData; // cache
    try {
      const res = await fetch("/static/gca.txt");
      if (!res.ok)
        throw new Error(
          "โหลด gca.txt ไม่ได้ (HTTP " + res.status + ")",
        );
      const text = await res.text();
      const lines = text.trim().split("\n");
      const result = {};
      for (const line of lines) {
        if (!line.trim() || line.startsWith("year")) continue;
        const parts = line.trim().split(/\s+/);
        const year = parseInt(parts[0], 10);
        const val = parseFloat(parts.slice(1).join("").replace(/,/g, ""));
        if (!isNaN(year) && !isNaN(val)) {
          result[year] = val;
        }
      }
      gcaData = result;
      console.log("GCA DATA:", gcaData);
      return gcaData;
    } catch (e) {
      alert(
        "ไม่สามารถโหลด gca.txt: " +
          e.message +
          "\nกรุณาวางไฟล์ gca.txt ไว้ในโฟลเดอร์ static/ เดียวกับ index.html",
      );
      return null;
    }
  }

  document
    .getElementById("toggleEdgarBtn")
    .addEventListener("click", async () => {
      const btn = document.getElementById("toggleEdgarBtn");
      const legend = document.getElementById("edgarLegend");
      if (!edgarVisible) {
        btn.innerHTML =
          '<i class="bi bi-x-circle me-1"></i>ซ่อนค่า EDGAR';
        btn.style.backgroundColor = "#6c757d";
        legend.style.display = "inline";
        await loadEdgarData();
      } else {
        btn.innerHTML =
          '<i class="bi bi-database-add me-1"></i>แสดงค่า EDGAR';
        btn.style.backgroundColor = "#fd7e14";
        legend.style.display = "none";
      }
      edgarVisible = !edgarVisible;
      drawTotalChart();
    });

  document
    .getElementById("toggleGcaBtn")
    .addEventListener("click", async () => {
      const btn = document.getElementById("toggleGcaBtn");
      const legend = document.getElementById("gcaLegend");
      if (!gcaVisible) {
        btn.innerHTML =
          '<i class="bi bi-x-circle me-1"></i>ซ่อนค่า GCA';
        btn.style.backgroundColor = "#6c757d";
        legend.style.display = "inline";
        await loadGCAData();
      } else {
        btn.innerHTML =
          '<i class="bi bi-database-add me-1"></i>แสดงค่า GCA';
        btn.style.backgroundColor = "#28a745";
        legend.style.display = "none";
      }
      gcaVisible = !gcaVisible;
      drawTotalChart();
    });

  provinceSearchInput.addEventListener("input", (e) =>
    filterCheckboxes(e.target.value),
  );
  useNewDataCheck.addEventListener("change", function () {
    fileInput.style.display = this.checked ? "block" : "none";
    if (!this.checked) fileInput.value = "";
  });

  uploadBtn.addEventListener("click", async (e) => {
    e.preventDefault();
    if (uploadInProgress) return;
    pageErrorDiv.style.display = "none";
    const useNew = useNewDataCheck.checked;
    let file = null;
    if (useNew) {
      file = fileInput.files[0];
      if (!file) {
        alert("กรุณาเลือกไฟล์ CSV สำหรับปี 2023 ขึ้นไป");
        return;
      }
    }
    let nYears = parseInt(nYearsInput.value, 10);
    if (isNaN(nYears) || nYears < 1) nYears = 1;
    if (nYears > 10) nYears = 10;
    nYearsInput.value = nYears;

    const formData = new FormData();
    if (useNew && file) formData.append("file", file);

    uploadInProgress = true;
    loadingDiv.style.display = "block";
    evalSection.style.display = "none";
    predSection.style.display = "none";
    uploadBtn.disabled = true;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180000);

    try {
      const startYear = startYearInput.value;
      const endYear = endYearInput.value;
      let queryParams = `n_years=${nYears}`;
      if (startYear) queryParams += `&start_year=${startYear}`;
      if (endYear) queryParams += `&end_year=${endYear}`;

      persistLog(`Sending request to /predict?${queryParams}`);
      const model = document.getElementById("modelSelect").value;
      const endpoint = model === "gnn" ? "/predict/gnn" : "/predict";
      const response = await fetch(`${endpoint}?${queryParams}`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      persistLog("Response received");

      const evalSummaryDiv = document.getElementById("evalSummary");
      if (data.evaluation_summary) {
        evalSummaryDiv.innerHTML = `<div class="row text-center"><div class="col-sm-6 mb-2"><span class="badge bg-success fs-6 p-2">MAPE</span><h3 class="mt-2 text-success">${(data.evaluation_summary.mape * 100).toFixed(2)}%</h3><small>Mean Absolute Percentage Error</small></div><div class="col-sm-6 mb-2"><span class="badge bg-primary fs-6 p-2">R²</span><h3 class="mt-2 text-primary">${data.evaluation_summary.r2.toFixed(4)}</h3><small>Coefficient of Determination</small></div></div>`;
      } else
        evalSummaryDiv.innerHTML =
          '<div class="alert alert-warning">ไม่สามารถคำนวณค่าประเมินโมเดล</div>';

      if (data.prediction && data.prediction.length) {
        currentPredictionData = data.prediction;
        const allPredValues = data.prediction.map((d) => d.preds);
        globalMaxPred = Math.max(...allPredValues);
        currentHistoricalData = data.historical || [];

        const predYears = [
          ...new Set(data.prediction.map((r) => r.year)),
        ].sort();
        const histYears = currentHistoricalData.length
          ? [...new Set(currentHistoricalData.map((r) => r.year))]
          : [];
        allYearsList = [...new Set([...histYears, ...predYears])].sort(
          (a, b) => a - b,
        );
        firstPredYearValue = Math.min(...predYears);

        displayTable("predTable", data.prediction, [
          "province",
          "year",
          "preds",
        ]);

        const yearTotals = {};
        data.prediction.forEach(
          (r) =>
            (yearTotals[r.year] =
              (yearTotals[r.year] || 0) + (Number(r.preds) || 0)),
        );
        const years = Object.keys(yearTotals).sort();
        document.getElementById("yearSummaryCards").innerHTML = years
          .map(
            (yr) =>
              `<div class="year-card"><div class="yr-label">ปี ${yr}</div><div class="yr-value">${yearTotals[yr].toLocaleString(undefined, { maximumFractionDigits: 0 })}</div><div class="yr-unit">ตัน CO₂</div></div>`,
          )
          .join("");
        const totalCO2 = Object.values(yearTotals).reduce(
          (s, v) => s + v,
          0,
        );
        document.getElementById("totalCo2Value").innerHTML =
          totalCO2.toLocaleString(undefined, {
            maximumFractionDigits: 0,
          });
        const yearRangeText =
          years.length === 1
            ? `${years[0]}`
            : `${years[0]} – ${years[years.length - 1]}`;
        document.getElementById("predictionYear").textContent =
          yearRangeText;
        document.getElementById("predictionYearRange").textContent =
          yearRangeText;

        const allProvinces = [
          ...new Set(data.prediction.map((r) => r.province)),
        ];
        renderProvinceCheckboxes(allProvinces);
        const top5Default = getTop5Provinces(data.prediction);
        setCheckboxesSelection(top5Default);
        redrawChart(top5Default);
        drawTotalChart();
        document.dispatchEvent(new Event("externalFactorsTrigger"));

        evalSection.style.display = "block";
        predSection.style.display = "block";
        showMapBtn();
      } else throw new Error("ไม่มีข้อมูลการทำนาย");
    } catch (error) {
      persistLog(`Error: ${error.message}`, "error");
      let msg =
        error.name === "AbortError"
          ? "การเชื่อมต่อใช้เวลานานเกินไป"
          : error.message.includes("Failed to fetch")
            ? "ไม่สามารถเชื่อมต่อ Backend ได้"
            : error.message;
      pageErrorDiv.style.display = "block";
      pageErrorDiv.innerText = msg;
      evalSection.style.display = "none";
      predSection.style.display = "none";
    } finally {
      uploadInProgress = false;
      loadingDiv.style.display = "none";
      uploadBtn.disabled = false;
    }
  });

  function displayTable(tableId, data, columns) {
    const tbody = document.querySelector(`#${tableId} tbody`);
    tbody.innerHTML = "";
    data.forEach((row) => {
      const tr = document.createElement("tr");
      columns.forEach((col) => {
        const td = document.createElement("td");
        let val = row[col];
        if (col === "preds") {
          val = Number(val).toLocaleString(undefined, {
            maximumFractionDigits: 0,
          });
        }
        td.textContent = val;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }

  window.addEventListener("beforeunload", (event) => {
    if (uploadInProgress) {
      event.preventDefault();
      event.returnValue = "กำลังประมวลผลข้อมูล กรุณารอสักครู่";
    }
  });
  persistLog("Page loaded");

  /* ══════════════════════════════════════════════════════ MAP PRESENTATION MODE ══════════════════════════════════════════════════════ */
  const mapPresBtn = document.getElementById("mapPresBtn");
  const mapPresOverlay = document.getElementById("mapPresOverlay");
  const mapPresClose = document.getElementById("mapPresClose");
  const tooltip = document.getElementById("mapTooltip");
  let mpSelectedYear = null;
  let thaiGeo = null;
  let mpPathMap = {};

  const EN2TH = {
    Bangkok: "กรุงเทพมหานคร",
    Krabi: "กระบี่",
    Kanchanaburi: "กาญจนบุรี",
    Kalasin: "กาฬสินธุ์",
    "Kamphaeng Phet": "กำแพงเพชร",
    "Khon Kaen": "ขอนแก่น",
    Chanthaburi: "จันทบุรี",
    Chachoengsao: "ฉะเชิงเทรา",
    Chonburi: "ชลบุรี",
    Chainat: "ชัยนาท",
    Chaiyaphum: "ชัยภูมิ",
    Chumphon: "ชุมพร",
    "Chiang Rai": "เชียงราย",
    "Chiang Mai": "เชียงใหม่",
    Trang: "ตรัง",
    Trat: "ตราด",
    Tak: "ตาก",
    "Nakhon Nayok": "นครนายก",
    "Nakhon Pathom": "นครปฐม",
    "Nakhon Phanom": "นครพนม",
    "Nakhon Ratchasima": "นครราชสีมา",
    "Nakhon Si Thammarat": "นครศรีธรรมราช",
    "Nakhon Sawan": "นครสวรรค์",
    Nonthaburi: "นนทบุรี",
    Narathiwat: "นราธิวาส",
    Nan: "น่าน",
    Buriram: "บุรีรัมย์",
    "Pathum Thani": "ปทุมธานี",
    "Prachuap Khiri Khan": "ประจวบคีรีขันธ์",
    "Prachin Buri": "ปราจีนบุรี",
    Pattani: "ปัตตานี",
    "Phra Nakhon Si Ayutthaya": "พระนครศรีอยุธยา",
    Phayao: "พะเยา",
    "Phang Nga": "พังงา",
    Phatthalung: "พัทลุง",
    Phichit: "พิจิตร",
    Phitsanulok: "พิษณุโลก",
    Phetchaburi: "เพชรบุรี",
    Phetchabun: "เพชรบูรณ์",
    Phrae: "แพร่",
    Phuket: "ภูเก็ต",
    "Maha Sarakham": "มหาสารคาม",
    Mukdahan: "มุกดาหาร",
    "Mae Hong Son": "แม่ฮ่องสอน",
    Yasothon: "ยโสธร",
    Yala: "ยะลา",
    "Roi Et": "ร้อยเอ็ด",
    Ranong: "ระนอง",
    Rayong: "ระยอง",
    Ratchaburi: "ราชบุรี",
    Lopburi: "ลพบุรี",
    Lampang: "ลำปาง",
    Lamphun: "ลำพูน",
    Loei: "เลย",
    Sisaket: "ศรีสะเกษ",
    "Sakon Nakhon": "สกลนคร",
    Songkhla: "สงขลา",
    Satun: "สตูล",
    "Samut Prakan": "สมุทรปราการ",
    "Samut Songkhram": "สมุทรสงคราม",
    "Samut Sakhon": "สมุทรสาคร",
    "Sa Kaeo": "สระแก้ว",
    Saraburi: "สระบุรี",
    "Sing Buri": "สิงห์บุรี",
    Sukhothai: "สุโขทัย",
    "Suphan Buri": "สุพรรณบุรี",
    "Surat Thani": "สุราษฎร์ธานี",
    Surin: "สุรินทร์",
    "Nong Khai": "หนองคาย",
    "Nong Bua Lamphu": "หนองบัวลำภู",
    "Ang Thong": "อ่างทอง",
    "Amnat Charoen": "อำนาจเจริญ",
    "Udon Thani": "อุดรธานี",
    Uttaradit: "อุตรดิตถ์",
    "Uthai Thani": "อุทัยธานี",
    "Ubon Ratchathani": "อุบลราชธานี",
    "Bueng Kan": "บึงกาฬ",
  };

  const API_TO_GEOJSON = {
    Bangkok: "Bangkok Metropolis",
    Chainat: "Chai Nat",
    Chonburi: "Chon Buri",
    Lopburi: "Lop Buri",
    Buriram: "Buri Ram",
    Sisaket: "Si Sa Ket",
    "Nong Bua Lamphu": "Nong Bua Lam Phu",
    "Phang Nga": "Phangnga",
    Chachoengsao: "Chachoengsao",
    Chanthaburi: "Chanthaburi",
    "Chiang Mai": "Chiang Mai",
    "Chiang Rai": "Chiang Rai",
    Chumphon: "Chumphon",
    Kalasin: "Kalasin",
    "Kamphaeng Phet": "Kamphaeng Phet",
    Kanchanaburi: "Kanchanaburi",
    "Khon Kaen": "Khon Kaen",
    Krabi: "Krabi",
    Lampang: "Lampang",
    Lamphun: "Lamphun",
    "Mae Hong Son": "Mae Hong Son",
    "Maha Sarakham": "Maha Sarakham",
    Mukdahan: "Mukdahan",
    "Nakhon Nayok": "Nakhon Nayok",
    "Nakhon Pathom": "Nakhon Pathom",
    "Nakhon Phanom": "Nakhon Phanom",
    "Nakhon Ratchasima": "Nakhon Ratchasima",
    "Nakhon Sawan": "Nakhon Sawan",
    "Nakhon Si Thammarat": "Nakhon Si Thammarat",
    Nan: "Nan",
    Narathiwat: "Narathiwat",
    Nonthaburi: "Nonthaburi",
    "Nong Khai": "Nong Khai",
    "Pathum Thani": "Pathum Thani",
    Pattani: "Pattani",
    Phayao: "Phayao",
    Phetchabun: "Phetchabun",
    Phetchaburi: "Phetchaburi",
    Phichit: "Phichit",
    Phitsanulok: "Phitsanulok",
    Phrae: "Phrae",
    "Phra Nakhon Si Ayutthaya": "Phra Nakhon Si Ayutthaya",
    Phuket: "Phuket",
    "Prachin Buri": "Prachin Buri",
    "Prachuap Khiri Khan": "Prachuap Khiri Khan",
    Ranong: "Ranong",
    Ratchaburi: "Ratchaburi",
    Rayong: "Rayong",
    "Roi Et": "Roi Et",
    "Sa Kaeo": "Sa Kaeo",
    "Sakon Nakhon": "Sakon Nakhon",
    "Samut Prakan": "Samut Prakan",
    "Samut Sakhon": "Samut Sakhon",
    "Samut Songkhram": "Samut Songkhram",
    Saraburi: "Saraburi",
    Satun: "Satun",
    "Sing Buri": "Sing Buri",
    Songkhla: "Songkhla",
    Sukhothai: "Sukhothai",
    "Suphan Buri": "Suphan Buri",
    "Surat Thani": "Surat Thani",
    Surin: "Surin",
    Tak: "Tak",
    Trang: "Trang",
    Trat: "Trat",
    "Ubon Ratchathani": "Ubon Ratchathani",
    "Udon Thani": "Udon Thani",
    "Uthai Thani": "Uthai Thani",
    Uttaradit: "Uttaradit",
    Yala: "Yala",
    Yasothon: "Yasothon",
    "Bueng Kan": "Bueng Kan",
  };

  function showMapBtn() {
    mapPresBtn.style.display = "flex";
  }
  mapPresBtn.addEventListener("click", openMapPres);
  mapPresClose.addEventListener("click", () =>
    mapPresOverlay.classList.remove("active"),
  );

  async function openMapPres() {
    if (!currentPredictionData.length) return;
    mapPresOverlay.classList.add("active");
    document.getElementById("mpTimestamp").textContent =
      "อัปเดต: " + new Date().toLocaleString("th-TH");
    if (!thaiGeo) await loadThaiGeo();
    const predYears = [
      ...new Set(currentPredictionData.map((d) => d.year)),
    ].sort();
    mpSelectedYear = predYears[0];
    renderYearTabs(predYears);
    renderMapYear(mpSelectedYear);
  }

  async function loadThaiGeo() {
    try {
      const res = await fetch(
        "https://raw.githubusercontent.com/apisit/thailand.json/master/thailand.json",
      );
      if (!res.ok) throw new Error("โหลด map ไม่ได้");
      const geo = await res.json();
      thaiGeo = geo.features;
      drawBaseMap();
    } catch (e) {
      console.error("GeoJSON load failed", e);
    }
  }

  /* ══════════════════════════════════════════════════════ EXTERNAL FACTORS CHART ══════════════════════════════════════════════════════ */

  const EXTERNAL_SECTORS = {
    energy_transport:      { label: "พลังงานและคมนาคม",      color: "#1565C0", icon: "bi-fuel-pump-fill" },
    industrial_processes:  { label: "กระบวนการอุตสาหกรรม",   color: "#E65100", icon: "bi-gear-fill" },
    agriculture:           { label: "เกษตรกรรม",              color: "#2E7D32", icon: "bi-tree-fill" },
    waste:                 { label: "ของเสีย",                 color: "#6A1B9A", icon: "bi-trash3-fill" },
    forestry_land_use:     { label: "ป่าไม้และการใช้ที่ดิน",  color: "#00695C", icon: "bi-tree" },
    total_excl_forestry:   { label: "รวม (ไม่รวมป่าไม้)",     color: "#B71C1C", icon: "bi-bar-chart-fill" },
    total_incl_forestry:   { label: "รวมทั้งหมด",             color: "#37474F", icon: "bi-globe2" },
  };

  let externalRawData = null;        // array of row objects from external.txt
  let externalChartInstance = null;
  let externalVisibleSectors = new Set(["energy_transport", "industrial_processes", "agriculture", "total_excl_forestry"]);

  // ── โหลด external.txt ─────────────────────────────────────────────────────
  async function loadExternalData() {
    if (externalRawData) return externalRawData;
    try {
      const res = await fetch("/static/external.txt");
      if (!res.ok) throw new Error("HTTP " + res.status);
      const text = await res.text();
      const lines = text.trim().split("\n");
      const header = lines[0].trim().split(",");
      externalRawData = lines.slice(1)
        .filter(l => l.trim())
        .map(l => {
          const parts = l.trim().split(",");
          const obj = {};
          header.forEach((h, i) => {
            obj[h.trim()] = h.trim() === "year" ? parseInt(parts[i], 10) : parseFloat(parts[i]);
          });
          return obj;
        });
      return externalRawData;
    } catch (e) {
      console.warn("[external] โหลดไม่ได้:", e.message);
      return null;
    }
  }

  // ── วาดกราฟ external (Stacked Bar + เส้น total) ──────────────────────────
  function drawExternalChart() {
    const canvas = document.getElementById("externalFactorsChart");
    if (!canvas || !externalRawData) return;
    const ctx = canvas.getContext("2d");

    const years = externalRawData.map(d => d.year);

    // ภาคส่วนที่ปล่อย CO₂ (bar บวก) — ไม่รวม forestry และ total
    const emissionKeys = ["energy_transport", "industrial_processes", "agriculture", "waste"];
    // ภาคดูดซับ (bar ลบ)
    const absorptionKey = "forestry_land_use";

    const datasets = [];

    // bars ภาคปล่อย — แต่ละ sector
    emissionKeys.forEach(key => {
      const meta = EXTERNAL_SECTORS[key];
      datasets.push({
        type: "bar",
        label: meta.label,
        data: externalRawData.map(d => d[key] != null ? d[key] / 1_000_000 : 0),
        backgroundColor: meta.color + "cc",
        borderColor: meta.color,
        borderWidth: 1,
        stack: "emissions",
        order: 2,
      });
    });

    // bar ภาคดูดซับ (ค่าติดลบ — ลงใต้แกน)
    datasets.push({
      type: "bar",
      label: EXTERNAL_SECTORS[absorptionKey].label + " (ดูดซับ)",
      data: externalRawData.map(d => d[absorptionKey] != null ? d[absorptionKey] / 1_000_000 : 0),
      backgroundColor: "#00897B99",
      borderColor: "#00695C",
      borderWidth: 1,
      stack: "absorption",
      order: 2,
    });

    // เส้น total รวมป่าไม้ (net)
    datasets.push({
      type: "line",
      label: "ยอดสุทธิ (รวมป่าไม้)",
      data: externalRawData.map(d => d["total_incl_forestry"] != null ? d["total_incl_forestry"] / 1_000_000 : null),
      borderColor: "#37474F",
      backgroundColor: "transparent",
      borderWidth: 3,
      pointRadius: 4,
      pointBackgroundColor: "#37474F",
      tension: 0.25,
      spanGaps: true,
      stack: undefined,
      order: 1,
    });

    // เส้น total ไม่รวมป่าไม้ (gross)
    datasets.push({
      type: "line",
      label: "รวม (ไม่รวมป่าไม้)",
      data: externalRawData.map(d => d["total_excl_forestry"] != null ? d["total_excl_forestry"] / 1_000_000 : null),
      borderColor: "#B71C1C",
      backgroundColor: "transparent",
      borderWidth: 2,
      borderDash: [5, 3],
      pointRadius: 3,
      pointBackgroundColor: "#B71C1C",
      tension: 0.25,
      spanGaps: true,
      stack: undefined,
      order: 1,
    });

    if (externalChartInstance) externalChartInstance.destroy();
    externalChartInstance = new Chart(ctx, {
      type: "bar",
      data: { labels: years, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        interaction: { intersect: false, mode: "index" },
        scales: {
          x: {
            stacked: true,
            title: { display: true, text: "ปี" },
          },
          y: {
            stacked: true,
            title: { display: true, text: "ล้านตัน CO₂" },
            ticks: { callback: v => v.toLocaleString(undefined, { maximumFractionDigits: 0 }) },
            grid: {
              color: ctx2 => ctx2.tick.value === 0 ? "rgba(0,0,0,0.35)" : "rgba(0,0,0,0.06)",
              lineWidth: ctx2 => ctx2.tick.value === 0 ? 2 : 1,
            },
          },
        },
        plugins: {
          legend: {
            position: "bottom",
            labels: { font: { family: "Sarabun, sans-serif" }, boxWidth: 14, padding: 10 },
          },
          tooltip: {
            callbacks: {
              label: c => {
                const v = c.raw;
                if (v == null) return null;
                const sign = v < 0 ? "" : "+";
                return `${c.dataset.label}: ${sign}${v.toLocaleString(undefined, { maximumFractionDigits: 2 })} ล้านตัน`;
              },
              footer: items => {
                const net = items.find(i => i.dataset.label === "ยอดสุทธิ (รวมป่าไม้)");
                if (!net || net.raw == null) return "";
                return `ยอดสุทธิ CO₂: ${net.raw.toLocaleString(undefined, { maximumFractionDigits: 2 })} ล้านตัน`;
              },
            },
          },
        },
      },
    });
  }

  // ── สร้าง KPI cards (toggle ใช้ legend Chart.js แทน) ──────────────────────
  function buildExternalUI() {
    // ซ่อน toggle bar (ใช้ legend chart แทน)
    const bar = document.getElementById("externalToggleBar");
    if (bar) bar.style.display = "none";

    // KPI cards 5 ใบ: 4 ปล่อย + 1 ดูดซับ
    const kpiContainer = document.getElementById("externalKpiCards");
    if (!kpiContainer || !externalRawData || !externalRawData.length) return;
    const latest = externalRawData[externalRawData.length - 1];
    const first  = externalRawData[0];
    const latestYear = latest.year;

    const kpiSectors = [
      { key: "energy_transport",     isAbsorb: false },
      { key: "industrial_processes", isAbsorb: false },
      { key: "agriculture",          isAbsorb: false },
      { key: "waste",                isAbsorb: false },
      { key: "forestry_land_use",    isAbsorb: true  },
    ];

    kpiContainer.innerHTML = kpiSectors.map(({ key, isAbsorb }) => {
      const meta = EXTERNAL_SECTORS[key];
      const val = latest[key];
      const valFirst = first[key];
      const pct = valFirst && valFirst !== 0 ? (((val - valFirst) / Math.abs(valFirst)) * 100).toFixed(1) : null;
      const sign = pct >= 0 ? "+" : "";
      // สำหรับ forestry: ค่าติดลบมากขึ้น = ดีขึ้น (ดูดซับมากขึ้น) → สีเขียว
      const pctColor = isAbsorb
        ? (pct < 0 ? "#2e7d32" : "#b71c1c")
        : (pct >= 0 ? "#b71c1c" : "#2e7d32");
      const displayVal = isAbsorb
        ? `${(Math.abs(val) / 1e6).toLocaleString(undefined, { maximumFractionDigits: 1 })} <small style="font-size:0.7rem;color:#888">ล้านตัน</small>`
        : `${(val / 1e6).toLocaleString(undefined, { maximumFractionDigits: 1 })} <small style="font-size:0.7rem;color:#888">ล้านตัน</small>`;
      const badge = isAbsorb
        ? `<span style="font-size:0.68rem;background:#e0f2f1;color:#00695C;border-radius:99px;padding:1px 8px;font-weight:600">🌿 ดูดซับ</span>`
        : `<span style="font-size:0.68rem;background:#fce4ec;color:#b71c1c;border-radius:99px;padding:1px 8px;font-weight:600">🏭 ปล่อย</span>`;
      return `
        <div class="col-6 col-md col-lg">
          <div class="stats-card" style="border-left:4px solid ${meta.color};">
            <i class="bi ${meta.icon} fs-3" style="color:${meta.color}"></i>
            <div class="mt-1">${badge}</div>
            <div class="mt-1 text-secondary" style="font-size:0.75rem">${meta.label}</div>
            <div class="fw-bold" style="font-size:1.1rem;color:${meta.color}">${displayVal}</div>
            ${pct != null ? `<div style="font-size:0.72rem;color:${pctColor}">${sign}${pct}% จากปี ${first.year}</div>` : ""}
            <div style="font-size:0.68rem;color:#aaa">ปี ${latestYear}</div>
          </div>
        </div>`;
    }).join("");
  }

  // ── โหลดและแสดงผลเมื่อ predSection แสดง ────────────────────────────────
  async function initExternalFactors() {
    await loadExternalData();
    if (!externalRawData) return;
    buildExternalUI();
    drawExternalChart();
  }

  // hook เข้ากับการแสดง predSection (เรียกหลัง drawTotalChart)
  const _origDrawTotalChart = drawTotalChart;
  // ไม่ override — เรียก initExternalFactors() ต่างหากจาก uploadBtn handler
  // แทรก initExternalFactors() หลัง drawTotalChart() ใน upload handler ด้านบน
  // (เพิ่มใน block ที่ predSection ถูก set visible)
  document.addEventListener("externalFactorsTrigger", initExternalFactors);

  function drawBaseMap() {
    const svg = d3.select("#thaiMapSvg");
    svg.selectAll("*").remove();
    mpPathMap = {};
    const projection = d3.geoMercator().fitSize([560, 980], {
      type: "FeatureCollection",
      features: thaiGeo,
    });
    const path = d3.geoPath().projection(projection);
    thaiGeo.forEach((feature) => {
      const nameTh = feature.properties.name || "";
      const nameEn = feature.properties.name_en || "";
      const el = svg
        .append("path")
        .datum(feature)
        .attr("d", path)
        .attr("data-name-th", nameTh)
        .attr("data-name-en", nameEn)
        .attr("fill", "#1a3d22");
      mpPathMap[nameTh] = el;
      if (nameEn) mpPathMap[nameEn] = el;
    });
  }

  function renderYearTabs(years) {
    const bar = document.getElementById("mpYearBar");
    bar.innerHTML = "";
    years.forEach((yr) => {
      const btn = document.createElement("button");
      btn.className =
        "mp-ytab" + (yr === mpSelectedYear ? " active" : "");
      btn.textContent = "ปี " + yr;
      btn.onclick = () => {
        mpSelectedYear = yr;
        bar
          .querySelectorAll(".mp-ytab")
          .forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        renderMapYear(yr);
      };
      bar.appendChild(btn);
    });
  }

  function getTotalForYear(targetYear) {
    const histTotal = currentHistoricalData
      .filter((d) => d.year === targetYear)
      .reduce((s, d) => s + (Number(d.actual) || 0), 0);
    const predTotal = currentPredictionData
      .filter((d) => d.year === targetYear)
      .reduce((s, d) => s + (Number(d.preds) || 0), 0);
    return histTotal + predTotal;
  }

  function renderMapYear(year) {
    const data = currentPredictionData.filter((d) => d.year === year);
    if (!data.length) return;
    const sorted = [...data].sort((a, b) => b.preds - a.preds);
    const max = globalMaxPred || 1;
    const colorScale = d3
      .scaleSequential()
      .domain([0, max])
      .interpolator(
        d3.interpolateRgbBasis([
          "#ffffe5", // 1. ต่ำสุด (เหลืองสว่างมาก)
          "#fff7bc", // 2.
          "#fee391", // 3. (เหลือง)
          "#fec44f", // 4.
          "#fe9929", // 5. ปานกลาง (ส้ม)
          "#ec7014", // 6.
          "#cc4c02", // 7. (แดงอมส้ม)
          "#993404", // 8.
          "#662506"  // 9. สูงสุด (น้ำตาลเข้ม/แดงเลือดหมู)
        ])
      );

    if (thaiGeo) {
      d3.selectAll("#thaiMapSvg path").attr("fill", "#1a3d22");
      data.forEach((d) => {
        let found = false;
        let geoKey = null;
        if (API_TO_GEOJSON[d.province]) {
          geoKey = API_TO_GEOJSON[d.province];
          if (mpPathMap[geoKey]) found = true;
        }
        if (!found) {
          geoKey = d.province;
          if (mpPathMap[geoKey]) found = true;
        }
        if (!found) {
          geoKey = EN2TH[d.province];
          if (mpPathMap[geoKey]) found = true;
        }
        if (!found) {
          const lower = d.province.toLowerCase();
          for (let [key, el] of Object.entries(mpPathMap)) {
            if (key.toLowerCase() === lower) {
              geoKey = key;
              found = true;
              break;
            }
          }
        }
        if (found && mpPathMap[geoKey]) {
          mpPathMap[geoKey]
            .attr("fill", colorScale(d.preds))
            .attr("data-preds", d.preds)
            .attr("data-province-en", d.province);
        }
      });

      d3.selectAll("#thaiMapSvg path")
        .on("mousemove", function (event) {
          const preds = +this.getAttribute("data-preds");
          const provEn =
            this.getAttribute("data-province-en") ||
            this.getAttribute("data-name-en");
          const provTh =
            EN2TH[provEn] ||
            this.getAttribute("data-name-th") ||
            provEn ||
            "–";
          if (!preds) {
            tooltip.style.display = "none";
            return;
          }
          const rank = sorted.findIndex((d) => d.province === provEn) + 1;
          document.getElementById("ttProv").textContent =
            provTh + (provEn ? ` (${provEn})` : "");
          document.getElementById("ttVal").textContent =
            (preds / 1e6).toLocaleString("th-TH", {
              maximumFractionDigits: 3,
            }) + " ล้านตัน CO₂";
          document.getElementById("ttRank").textContent = rank
            ? `อันดับที่ ${rank} จาก ${data.length} จังหวัด`
            : "";
          tooltip.style.display = "block";
          tooltip.style.left = event.clientX + 14 + "px";
          tooltip.style.top = event.clientY - 10 + "px";
        })
        .on("mouseleave", () => {
          tooltip.style.display = "none";
        });
    }

    const total = data.reduce((s, d) => s + (Number(d.preds) || 0), 0);
    const prevTotal = getTotalForYear(year - 1);
    let yoyHtml = "–";
    if (prevTotal > 0) {
      const pct = (((total - prevTotal) / prevTotal) * 100).toFixed(2);
      const sign = pct >= 0 ? "+" : "";
      const col = pct >= 0 ? "#ef9a9a" : "#a5d6a7";
      yoyHtml = `<span style="color:${col}">${sign}${pct}%</span>`;
    }
    document.getElementById("mpKpiStrip").innerHTML = `
<div class="mp-kpi"><div class="mk-label">CO₂ รวม (ปี ${year})</div><div class="mk-val">${(total / 1e6).toLocaleString("th-TH", { maximumFractionDigits: 2 })}</div><div class="mk-unit">ล้านตัน</div></div>
<div class="mp-kpi"><div class="mk-label">เทียบปีก่อน</div><div class="mk-val">${yoyHtml}</div><div class="mk-unit">YoY</div></div>
<div class="mp-kpi"><div class="mk-label">สูงสุด</div><div class="mk-val" style="font-size:.95rem">${EN2TH[sorted[0]?.province] || sorted[0]?.province || "–"}</div><div class="mk-unit">${((sorted[0]?.preds || 0) / 1e6).toFixed(2)} M ตัน</div></div>
<div class="mp-kpi"><div class="mk-label">จังหวัดทั้งหมด</div><div class="mk-val">${data.length}</div><div class="mk-unit">จังหวัด</div></div>`;

    document.getElementById("mpRankScroll").innerHTML = sorted
      .map((d, i) => {
        const thName = EN2TH[d.province] || d.province;
        const pct = ((d.preds / max) * 100).toFixed(1);
        const badgeCls =
          i === 0 ? "r1" : i === 1 ? "r2" : i === 2 ? "r3" : "rn";
        return `<div class="mp-rank-item" data-prov-en="${d.province}"><div class="mp-rank-num ${badgeCls}">${i + 1}</div><div class="mp-rank-name">${thName}</div><div class="mp-rank-barwrap"><div class="mp-rank-bar" style="width:${pct}%"></div></div><div class="mp-rank-val">${(d.preds / 1e6).toFixed(2)}M</div></div>`;
      })
      .join("");

    document.querySelectorAll(".mp-rank-item").forEach((item) => {
      item.addEventListener("mouseenter", () => {
        const en = item.getAttribute("data-prov-en");
        let el = null;
        let geoKey = API_TO_GEOJSON[en];
        if (geoKey && mpPathMap[geoKey]) el = mpPathMap[geoKey];
        if (!el && mpPathMap[en]) el = mpPathMap[en];
        if (!el) {
          const th = EN2TH[en];
          if (th && mpPathMap[th]) el = mpPathMap[th];
        }
        if (!el) {
          const lower = en.toLowerCase();
          for (let [key, val] of Object.entries(mpPathMap)) {
            if (key.toLowerCase() === lower) {
              el = val;
              break;
            }
          }
        }
        if (el) {
          el.style("filter", "brightness(1.8)")
            .style("stroke", "#c9a03d")
            .style("stroke-width", "2");
        }
      });
      item.addEventListener("mouseleave", () => {
        const en = item.getAttribute("data-prov-en");
        let el = null;
        let geoKey = API_TO_GEOJSON[en];
        if (geoKey && mpPathMap[geoKey]) el = mpPathMap[geoKey];
        if (!el && mpPathMap[en]) el = mpPathMap[en];
        if (!el) {
          const th = EN2TH[en];
          if (th && mpPathMap[th]) el = mpPathMap[th];
        }
        if (!el) {
          const lower = en.toLowerCase();
          for (let [key, val] of Object.entries(mpPathMap)) {
            if (key.toLowerCase() === lower) {
              el = val;
              break;
            }
          }
        }
        if (el) {
          el.style("filter", "")
            .style("stroke", "#1a3d22")
            .style("stroke-width", ".5");
        }
      });
    });
  }
})();