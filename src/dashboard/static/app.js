/**
 * FinRL-X-MT5: Council Terminal Frontend Engine
 * Handles 60fps TradingView canvas charting, live WebSocket streaming,
 * reactive SVG prop firm risk meters, and interactive Chain-of-Thought rendering.
 */

let chartInstance = null;
let candleSeries = null;
let volumeSeries = null;
let activeSymbol = "NAS100.x";
let wsConnection = null;

// Circumference of SVG gauge (r=42 -> 2 * PI * 42 ≈ 263.89)
const GAUGE_CIRCUMFERENCE = 263.89;

document.addEventListener("DOMContentLoaded", () => {
    initChart();
    initTheme();
    initDonationModal();
    initWebSocket();
    setupEventListeners();

    // Initial Data Hydration
    refreshAllData();

    // Auto-polling background timers
    setInterval(fetchAccount, 3000);
    setInterval(fetchPositions, 5000);
});

// ─── 1. TradingView Lightweight Chart Initialization ─────────────────────────
function initChart() {
    const container = document.getElementById("chartContainer");
    if (!container) return;

    chartInstance = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: {
            background: { color: "#07090e" },
            textColor: "#94a3b8",
            fontSize: 11,
            fontFamily: "'JetBrains Mono', monospace",
        },
        grid: {
            vertLines: { color: "rgba(255, 255, 255, 0.04)" },
            horzLines: { color: "rgba(255, 255, 255, 0.04)" },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: { color: "rgba(0, 242, 254, 0.4)", width: 1, style: 2 },
            horzLine: { color: "rgba(0, 242, 254, 0.4)", width: 1, style: 2 },
        },
        rightPriceScale: {
            borderColor: "rgba(255, 255, 255, 0.08)",
            scaleMargins: { top: 0.1, bottom: 0.2 },
        },
        timeScale: {
            borderColor: "rgba(255, 255, 255, 0.08)",
            timeVisible: true,
            secondsVisible: false,
        },
    });

    // Candlestick Series
    candleSeries = chartInstance.addCandlestickSeries({
        upColor: "#10b981",
        downColor: "#f43f5e",
        borderUpColor: "#10b981",
        borderDownColor: "#f43f5e",
        wickUpColor: "#10b981",
        wickDownColor: "#f43f5e",
    });

    // Volume Histogram Series
    volumeSeries = chartInstance.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "",
        scaleMargins: { top: 0.82, bottom: 0 },
    });

    // Crosshair hover readout
    chartInstance.subscribeCrosshairMove((param) => {
        if (!param || !param.time || !param.seriesPrices) return;
        const price = param.seriesPrices.get(candleSeries);
        const vol = param.seriesPrices.get(volumeSeries);
        if (price) {
            document.getElementById("roOpen").innerText = price.open.toFixed(2);
            document.getElementById("roHigh").innerText = price.high.toFixed(2);
            document.getElementById("roLow").innerText = price.low.toFixed(2);
            document.getElementById("roClose").innerText = price.close.toFixed(2);
            if (vol !== undefined) document.getElementById("roVol").innerText = vol.toLocaleString();
        }
    });

    // Resize listener
    window.addEventListener("resize", () => {
        if (chartInstance && container) {
            chartInstance.applyOptions({
                width: container.clientWidth,
                height: container.clientHeight,
            });
        }
    });
}

// ─── 2. Data Fetchers ────────────────────────────────────────────────────────
async function refreshAllData() {
    await fetchAccount();
    await fetchChartBars(activeSymbol);
    await fetchPositions();
    await fetchCouncilDecision(activeSymbol);
}

async function fetchAccount() {
    try {
        const res = await fetch("/api/account");
        if (!res.ok) return;
        const data = await res.json();

        if (data.connected) {
            document.getElementById("liveDot").style.background = "#10b981";
            document.getElementById("brokerServer").innerText = data.server || "Connected";
            document.getElementById("accountLogin").innerText = `#${data.login}`;
            document.getElementById("accountEquity").innerText = `$${data.equity.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
            document.getElementById("accountBalance").innerText = `$${data.balance.toLocaleString(undefined, {minimumFractionDigits: 2})}`;
            
            const profitEl = document.getElementById("accountProfit");
            profitEl.innerText = `${data.profit >= 0 ? "+" : ""}$${data.profit.toFixed(2)}`;
            profitEl.style.color = data.profit >= 0 ? "var(--accent-emerald)" : "var(--accent-rose)";

            // Update Prop Firm Guardian Gauges
            if (data.prop_firm) {
                updateGuardianGauges(data.prop_firm);
            }
        }
    } catch (err) {
        console.warn("Account poll error:", err);
    }
}

function updateGuardianGauges(pf) {
    // 1. Daily Drawdown
    const dailyPct = pf.daily_drawdown_pct;
    const dailyLimit = pf.daily_limit_pct; // 2.5%
    const dailyRatio = Math.min(1.0, dailyPct / dailyLimit);
    const dailyOffset = GAUGE_CIRCUMFERENCE * (1.0 - dailyRatio);
    
    const dailyFill = document.getElementById("gaugeDailyFill");
    dailyFill.style.strokeDashoffset = dailyOffset;
    document.getElementById("gaugeDailyVal").innerText = `${dailyPct.toFixed(1)}%`;
    
    const dailyStatus = document.getElementById("gaugeDailyStatus");
    dailyStatus.innerText = `$${pf.daily_drawdown_usd.toFixed(2)} / $${((pf.daily_limit_pct / 100) * 10000).toFixed(0)}`;
    if (dailyRatio > 0.7) {
        dailyFill.style.stroke = "var(--accent-rose)";
        dailyStatus.className = "gauge-status text-rose";
    } else if (dailyRatio > 0.4) {
        dailyFill.style.stroke = "var(--accent-amber)";
        dailyStatus.className = "gauge-status text-amber";
    } else {
        dailyFill.style.stroke = "var(--accent-emerald)";
        dailyStatus.className = "gauge-status text-emerald";
    }

    // 2. Total Drawdown
    const totalPct = pf.total_drawdown_pct;
    const totalLimit = pf.total_limit_pct; // 4.0%
    const totalRatio = Math.min(1.0, totalPct / totalLimit);
    const totalOffset = GAUGE_CIRCUMFERENCE * (1.0 - totalRatio);

    const totalFill = document.getElementById("gaugeTotalFill");
    totalFill.style.strokeDashoffset = totalOffset;
    document.getElementById("gaugeTotalVal").innerText = `${totalPct.toFixed(1)}%`;
    document.getElementById("gaugeTotalStatus").innerText = `$${pf.total_drawdown_usd.toFixed(2)} / $${((pf.total_limit_pct / 100) * 10000).toFixed(0)}`;

    // 3. Target Progress
    const progPct = pf.profit_progress_pct;
    const progRatio = Math.min(1.0, progPct / 100.0);
    const progOffset = GAUGE_CIRCUMFERENCE * (1.0 - progRatio);

    const targetFill = document.getElementById("gaugeTargetFill");
    targetFill.style.strokeDashoffset = progOffset;
    document.getElementById("gaugeTargetVal").innerText = `${progPct.toFixed(0)}%`;
}

async function fetchChartBars(symbol) {
    try {
        document.getElementById("chartSymbolTitle").innerText = symbol;
        const res = await fetch(`/api/bars?symbol=${symbol}&count=160`);
        if (!res.ok) return;
        const data = await res.json();

        if (data.bars && data.bars.length > 0) {
            candleSeries.setData(data.bars.map(b => ({
                time: b.time,
                open: b.open,
                high: b.high,
                low: b.low,
                close: b.close
            })));

            volumeSeries.setData(data.bars.map(b => ({
                time: b.time,
                value: b.volume,
                color: b.close >= b.open ? "rgba(16, 185, 129, 0.35)" : "rgba(244, 63, 94, 0.35)",
            })));

            chartInstance.timeScale().fitContent();

            // Set live quote
            if (data.quote) {
                document.getElementById("quoteBid").innerText = data.quote.bid.toFixed(2);
                document.getElementById("quoteAsk").innerText = data.quote.ask.toFixed(2);
                document.getElementById("quoteSpread").innerText = `Spr: ${data.quote.spread.toFixed(2)}`;
            }
        }
    } catch (err) {
        console.error("Failed to load bars:", err);
    }
}

async function fetchCouncilDecision(symbol) {
    appendCoTLine(`[${new Date().toLocaleTimeString()}] Convening Council specialists for ${symbol}...`, "log-system");
    try {
        const res = await fetch(`/api/council/decision?symbol=${symbol}`);
        if (!res.ok) {
            appendCoTLine(`[${new Date().toLocaleTimeString()}] ⚠️ Could not fetch deliberation for ${symbol}`, "log-withheld");
            return;
        }
        const cot = await res.json();

        renderDeliberation(cot);
    } catch (err) {
        console.error("Council error:", err);
    }
}

function renderDeliberation(cot) {
    // 1. Verdict Banner
    const banner = document.getElementById("verdictBanner");
    const icon = document.getElementById("verdictIcon");
    const title = document.getElementById("verdictTitle");
    const desc = document.getElementById("verdictDesc");

    if (cot.is_tradeable) {
        banner.className = "verdict-banner verdict-approved";
        icon.innerText = "✅";
        title.innerText = `${cot.direction} — TRADE APPROVED`;
        desc.innerText = cot.verdict;
    } else {
        banner.className = "verdict-banner verdict-withheld";
        icon.innerText = "🛡️";
        title.innerText = "WITHHELD — CAPITAL PRESERVATION";
        desc.innerText = cot.verdict;
    }

    // 2. Thermometer
    const sig = cot.consensus_signal; // [-1.0, 1.0]
    document.getElementById("consensusSignalVal").innerText = sig > 0 ? `+${sig.toFixed(3)}` : sig.toFixed(3);
    const bar = document.getElementById("consensusSignalBar");

    if (sig >= 0) {
        const widthPct = (sig / 1.0) * 50;
        bar.style.left = "50%";
        bar.style.width = `${widthPct}%`;
        bar.style.background = "var(--accent-emerald)";
    } else {
        const widthPct = (Math.abs(sig) / 1.0) * 50;
        bar.style.left = `${50 - widthPct}%`;
        bar.style.width = `${widthPct}%`;
        bar.style.background = "var(--accent-rose)";
    }

    // 3. Mini Pills
    document.getElementById("cotRegime").innerText = cot.regime;
    document.getElementById("cotConfidence").innerText = `${(cot.confidence * 100).toFixed(1)}%`;
    document.getElementById("cotRR").innerText = cot.expected_rr.toFixed(2);
    document.getElementById("cotKelly").innerText = `${cot.position_size.toFixed(2)}x`;

    // 4. Specialist Cards
    const specList = document.getElementById("specialistsList");
    specList.innerHTML = "";

    cot.steps.forEach(step => {
        const row = document.createElement("div");
        row.className = "specialist-row";

        const sigColor = step.signal > 0.05 ? "text-emerald" : step.signal < -0.05 ? "text-rose" : "text-cyan";
        const weightPct = ((step.weight || 0.2) * 100).toFixed(0);

        row.innerHTML = `
            <div class="spec-header">
                <span class="spec-name">${step.icon} ${step.expert}</span>
                <div class="spec-badges">
                    <span class="badge-sig ${sigColor}">SIG: ${step.signal > 0 ? "+" : ""}${step.signal.toFixed(2)}</span>
                    <span class="badge-weight">W: ${weightPct}%</span>
                </div>
            </div>
            <div class="spec-narrative">${step.narrative}</div>
        `;
        specList.appendChild(row);
    });

    // 5. Stream Chain of Thought to Terminal
    appendCoTLine(`------------------------------------------------------------`, "log-system");
    appendCoTLine(`[${new Date().toLocaleTimeString()}] 🏛️ DELIBERATION VERDICT FOR ${cot.symbol}:`, "log-check");
    cot.steps.forEach(s => {
        appendCoTLine(`▶ ${s.icon} ${s.expert}: ${s.narrative}`, "log-system");
    });
    cot.checkpoints.forEach(cp => {
        const status = cp.passed ? "✅ PASS" : "❌ FAIL";
        appendCoTLine(`  ${status} [${cp.name}]: ${cp.value}`, cp.passed ? "log-approved" : "log-withheld");
    });
    appendCoTLine(`VERDICT: ${cot.verdict}`, cot.is_tradeable ? "log-approved" : "log-withheld");
}

function appendCoTLine(text, cssClass = "log-system") {
    const terminal = document.getElementById("cotTerminal");
    if (!terminal) return;

    const line = document.createElement("div");
    line.className = `cot-line ${cssClass}`;
    line.innerText = text;
    terminal.appendChild(line);

    // Keep scroll at bottom
    terminal.scrollTop = terminal.scrollHeight;

    // Prune old lines if exceeding 200
    if (terminal.children.length > 200) {
        terminal.removeChild(terminal.firstChild);
    }
}

async function fetchPositions() {
    try {
        const res = await fetch("/api/positions");
        if (!res.ok) return;
        const data = await res.json();

        const countBadge = document.getElementById("openPosCount");
        const tbody = document.getElementById("positionsTbody");

        if (data.open_positions && data.open_positions.length > 0) {
            countBadge.innerText = `${data.open_positions.length} OPEN`;
            countBadge.style.background = "rgba(16, 185, 129, 0.15)";
            countBadge.style.color = "var(--accent-emerald)";

            tbody.innerHTML = data.open_positions.map(p => `
                <tr>
                    <td>#${p.ticket}</td>
                    <td class="font-bold">${p.symbol}</td>
                    <td class="${p.type === 'BUY' ? 'text-emerald' : 'text-rose'} font-bold">${p.type}</td>
                    <td>${p.volume.toFixed(2)}</td>
                    <td>${p.price_open.toFixed(2)}</td>
                    <td>${p.price_current.toFixed(2)}</td>
                    <td>${p.sl ? p.sl.toFixed(2) : '-'}</td>
                    <td>${p.tp ? p.tp.toFixed(2) : '-'}</td>
                    <td class="${p.profit >= 0 ? 'text-emerald' : 'text-rose'} font-bold">${p.profit >= 0 ? '+' : ''}$${p.profit.toFixed(2)}</td>
                </tr>
            `).join("");
        } else {
            countBadge.innerText = "0 OPEN";
            countBadge.style.background = "var(--bg-surface-elevated)";
            countBadge.style.color = "var(--text-muted)";
            tbody.innerHTML = `<tr><td colspan="9" class="empty-state">No open positions. Council waiting for next high-conviction setup.</td></tr>`;
        }
    } catch (err) {
        console.warn("Positions poll error:", err);
    }
}

// ─── 3. WebSocket Real-Time Stream ──────────────────────────────────────────
function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

    try {
        wsConnection = new WebSocket(wsUrl);

        wsConnection.onopen = () => {
            appendCoTLine(`[${new Date().toLocaleTimeString()}] Live WebSocket stream connected to MT5 kernel.`, "log-approved");
        };

        wsConnection.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === "HEARTBEAT" && msg.symbol === activeSymbol) {
                document.getElementById("quoteBid").innerText = msg.bid.toFixed(2);
                document.getElementById("quoteAsk").innerText = msg.ask.toFixed(2);
                document.getElementById("quoteSpread").innerText = `Spr: ${msg.spread.toFixed(1)}`;
            }
        };

        wsConnection.onclose = () => {
            setTimeout(initWebSocket, 3000);
        };
    } catch (e) {
        console.warn("WebSocket init error:", e);
    }
}

// ─── 4. Event Listeners ─────────────────────────────────────────────────────
function setupEventListeners() {
    const sel = document.getElementById("symbolSelect");
    if (sel) {
        sel.addEventListener("change", (e) => {
            activeSymbol = e.target.value;
            fetchChartBars(activeSymbol);
            fetchCouncilDecision(activeSymbol);
        });
    }

    const btnDelib = document.getElementById("btnDeliberate");
    if (btnDelib) {
        btnDelib.addEventListener("click", () => {
            fetchCouncilDecision(activeSymbol);
        });
    }

    const btnRef = document.getElementById("btnRefreshChart");
    if (btnRef) {
        btnRef.addEventListener("click", () => {
            fetchChartBars(activeSymbol);
            fetchAccount();
        });
    }
}

// ─── 5. Theme Management (Dark / White Mode) ─────────────────────────────────
function initTheme() {
    const savedTheme = localStorage.getItem("finrl_theme") || "dark";
    applyTheme(savedTheme);

    const themeBtn = document.getElementById("themeToggle");
    if (themeBtn) {
        themeBtn.addEventListener("click", () => {
            const currentTheme = document.body.classList.contains("light-mode") ? "light" : "dark";
            const newTheme = currentTheme === "dark" ? "light" : "dark";
            applyTheme(newTheme);
        });
    }
}

function applyTheme(theme) {
    const icon = document.getElementById("themeIcon");
    const label = document.getElementById("themeLabel");

    if (theme === "light") {
        document.body.classList.add("light-mode");
        if (icon) icon.innerText = "🌙";
        if (label) label.innerText = "DARK";
        localStorage.setItem("finrl_theme", "light");

        // Update TradingView Chart for Light Mode
        if (chartInstance) {
            chartInstance.applyOptions({
                layout: {
                    background: { color: "#ffffff" },
                    textColor: "#475569",
                },
                grid: {
                    vertLines: { color: "rgba(15, 23, 42, 0.05)" },
                    horzLines: { color: "rgba(15, 23, 42, 0.05)" },
                },
                rightPriceScale: {
                    borderColor: "rgba(15, 23, 42, 0.08)",
                },
                timeScale: {
                    borderColor: "rgba(15, 23, 42, 0.08)",
                },
            });
        }
    } else {
        document.body.classList.remove("light-mode");
        if (icon) icon.innerText = "☀️";
        if (label) label.innerText = "LIGHT";
        localStorage.setItem("finrl_theme", "dark");

        // Update TradingView Chart for Dark Mode
        if (chartInstance) {
            chartInstance.applyOptions({
                layout: {
                    background: { color: "#07090e" },
                    textColor: "#94a3b8",
                },
                grid: {
                    vertLines: { color: "rgba(255, 255, 255, 0.04)" },
                    horzLines: { color: "rgba(255, 255, 255, 0.04)" },
                },
                rightPriceScale: {
                    borderColor: "rgba(255, 255, 255, 0.08)",
                },
                timeScale: {
                    borderColor: "rgba(255, 255, 255, 0.08)",
                },
            });
        }
    }
}

// ─── 6. Support & Donation Modal ──────────────────────────────────────────────
function initDonationModal() {
    const donateBtn = document.getElementById("donateBtn");
    const modal = document.getElementById("donateModal");
    const closeBtn = document.getElementById("closeDonateModal");
    const copyBtn = document.getElementById("btnCopyAddress");
    const copyText = document.getElementById("copyText");
    const addrInput = document.getElementById("trc20Address");

    if (donateBtn && modal) {
        donateBtn.addEventListener("click", () => {
            modal.style.display = "flex";
        });
    }

    if (closeBtn && modal) {
        closeBtn.addEventListener("click", () => {
            modal.style.display = "none";
        });
    }

    if (modal) {
        modal.addEventListener("click", (e) => {
            if (e.target === modal) {
                modal.style.display = "none";
            }
        });
    }

    if (copyBtn && addrInput) {
        copyBtn.addEventListener("click", () => {
            addrInput.select();
            addrInput.setSelectionRange(0, 99999);
            navigator.clipboard.writeText(addrInput.value).then(() => {
                if (copyText) {
                    copyText.innerText = "COPIED! ✅";
                    copyBtn.style.background = "#10b981";
                    setTimeout(() => {
                        copyText.innerText = "COPY";
                        copyBtn.style.background = "";
                    }, 2000);
                }
            }).catch(() => {
                document.execCommand("copy");
                if (copyText) {
                    copyText.innerText = "COPIED! ✅";
                    setTimeout(() => {
                        copyText.innerText = "COPY";
                    }, 2000);
                }
            });
        });
    }
}
