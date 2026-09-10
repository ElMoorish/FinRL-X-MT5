/**
 * FinRL-X-MT5: Council Terminal Frontend Engine
 * Handles 60fps TradingView canvas charting, live WebSocket streaming,
 * reactive SVG prop firm risk meters, interactive Chain-of-Thought rendering,
 * modular card controls, slide-over layout customizer, and deals history.
 */

let chartInstance = null;
let candleSeries = null;
let volumeSeries = null;
let h1EmaSeries = null;
let activeSymbol = "NAS100.x";
let activeTimeframe = "M5";
let showH1Ema = true;
let currentH1EmaVal = 0.0;
let wsConnection = null;
let activeTab = "open";

// Circumference of SVG gauge (r=42 -> 2 * PI * 42 ≈ 263.89)
const GAUGE_CIRCUMFERENCE = 263.89;

// Layout Presets Definition
const LAYOUT_PRESETS = {
    default: {
        visible: ["cardChart", "cardGuardian", "cardPositions", "cardConsensus", "cardSpecialists", "cardCot"],
        collapsed: []
    },
    prop_trader: {
        visible: ["cardChart", "cardGuardian", "cardPositions", "cardConsensus"],
        collapsed: []
    },
    ai_researcher: {
        visible: ["cardChart", "cardConsensus", "cardSpecialists", "cardCot"],
        collapsed: []
    },
    scalper: {
        visible: ["cardChart", "cardPositions", "cardConsensus"],
        collapsed: []
    }
};

document.addEventListener("DOMContentLoaded", () => {
    initChart();
    initTheme();
    initDonationModal();
    initWebSocket();
    initTimeframeSelector();
    initPositionsTabs();
    initCardControls();
    initLayoutDrawer();
    setupEventListeners();

    // Initial Data Hydration
    refreshAllData();

    // Auto-polling background timers
    setInterval(fetchAccount, 3000);
    setInterval(fetchPositions, 4000);
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

    // H1 Macro Trend Governor Line Overlay (EMA 50)
    h1EmaSeries = chartInstance.addLineSeries({
        color: "#f59e0b",
        lineWidth: 2,
        lineStyle: 2, // Dashed
        title: "H1 EMA50",
        priceLineVisible: true,
        lastValueVisible: true,
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
    await fetchChartBars(activeSymbol, activeTimeframe);
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

async function fetchChartBars(symbol, tf = activeTimeframe) {
    try {
        activeTimeframe = tf;
        document.getElementById("chartSymbolTitle").innerText = symbol;
        const res = await fetch(`/api/bars?symbol=${symbol}&tf=${tf}&count=160`);
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

            // Update H1 Macro Trend Governor line overlay
            if (data.h1_ema > 0 && showH1Ema) {
                currentH1EmaVal = data.h1_ema;
                const emaPoints = data.bars.map(b => ({
                    time: b.time,
                    value: data.h1_ema,
                }));
                h1EmaSeries.setData(emaPoints);
                h1EmaSeries.applyOptions({ visible: true });
            } else {
                h1EmaSeries.applyOptions({ visible: false });
            }

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

async function fetchCouncilDecision(symbol, force = false) {
    appendCoTLine(`[${new Date().toLocaleTimeString()}] ${force ? 'Force refreshing' : 'Synchronizing'} Council deliberation for ${symbol}...`, "log-system");
    try {
        const res = await fetch(`/api/council/decision?symbol=${symbol}&force=${force}`);
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
        title.innerText = `${cot.direction} — HIGH-CONVICTION TRADE APPROVED`;
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
    const h1El = document.getElementById("cotH1Trend");
    if (h1El) {
        h1El.innerText = cot.h1_trend || "ALIGNED";
        if (cot.h1_trend && cot.h1_trend.includes("BULLISH")) {
            h1El.className = "pill-value font-bold text-emerald";
        } else if (cot.h1_trend && cot.h1_trend.includes("BEARISH")) {
            h1El.className = "pill-value font-bold text-rose";
        }
    }

    // 4. Specialist Cards
    const specList = document.getElementById("specialistsList");
    specList.innerHTML = "";

    cot.steps.forEach(step => {
        const row = document.createElement("div");
        row.className = "specialist-row";

        const sigColor = step.signal > 0.05 ? "text-emerald" : step.signal < -0.05 ? "text-rose" : "text-cyan";
        const weightPct = ((step.weight || 0.2) * 100).toFixed(1);
        const contribStr = step.contribution !== undefined ? 
            `<span class="badge-contrib font-mono">Contrib: ${step.contribution > 0 ? "+" : ""}${step.contribution.toFixed(3)}</span>` : "";

        row.innerHTML = `
            <div class="spec-header">
                <span class="spec-name">${step.icon} ${step.expert}</span>
                <div class="spec-badges">
                    <span class="badge-sig ${sigColor}">SIG: ${step.signal > 0 ? "+" : ""}${step.signal.toFixed(2)}</span>
                    <span class="badge-weight">W: ${weightPct}%</span>
                    ${contribStr}
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

        // 1. Open Positions
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
            countBadge.innerText = "0";
            tbody.innerHTML = `<tr><td colspan="9" class="empty-state">No open positions. Council waiting for next high-conviction setup.</td></tr>`;
        }

        // 2. Closed Deals History
        const dealsTbody = document.getElementById("dealsTbody");
        if (dealsTbody && data.recent_deals && data.recent_deals.length > 0) {
            dealsTbody.innerHTML = data.recent_deals.map(d => `
                <tr>
                    <td>#${d.ticket}</td>
                    <td class="font-mono text-muted">${d.time.split(' ')[1] || d.time}</td>
                    <td class="font-bold">${d.symbol}</td>
                    <td class="${d.type === 'BUY' ? 'text-emerald' : 'text-rose'} font-bold">${d.type}</td>
                    <td>${d.volume.toFixed(2)}</td>
                    <td>${d.price.toFixed(2)}</td>
                    <td class="text-muted font-mono">$${(d.commission || 0).toFixed(2)}</td>
                    <td class="${d.profit >= 0 ? 'text-emerald' : 'text-rose'} font-bold">${d.profit >= 0 ? '+' : ''}$${d.profit.toFixed(2)}</td>
                </tr>
            `).join("");
        } else if (dealsTbody) {
            dealsTbody.innerHTML = `<tr><td colspan="8" class="empty-state">No closed deals in the last 7 days.</td></tr>`;
        }

        // 3. Session Performance Summary Stats
        if (data.session_stats) {
            const ss = data.session_stats;
            const pnlEl = document.getElementById("sessPnl");
            const wrEl = document.getElementById("sessWr");
            const trEl = document.getElementById("sessTrades");

            if (pnlEl) {
                pnlEl.innerText = `${ss.today_pnl >= 0 ? "+" : ""}$${ss.today_pnl.toFixed(2)}`;
                pnlEl.className = ss.today_pnl >= 0 ? "text-emerald" : "text-rose";
            }
            if (wrEl) wrEl.innerText = `${ss.today_win_rate.toFixed(1)}%`;
            if (trEl) trEl.innerText = `${ss.today_trades}`;
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
            } else if (msg.type === "COUNCIL_DECISION" && msg.symbol === activeSymbol) {
                appendCoTLine(`[${new Date().toLocaleTimeString()}] ⚡ Live Council broadcast received (${msg.cot.source || 'LIVE'}).`, "log-check");
                renderDeliberation(msg.cot);
            }
        };

        wsConnection.onclose = () => {
            setTimeout(initWebSocket, 3000);
        };
    } catch (e) {
        console.warn("WebSocket init error:", e);
    }
}

// ─── 4. Timeframe Switcher & Chart Controls ───────────────────────────────────
function initTimeframeSelector() {
    const tfBtns = document.querySelectorAll(".btn-tf");
    tfBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tfBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            activeTimeframe = btn.dataset.tf;
            fetchChartBars(activeSymbol, activeTimeframe);
        });
    });

    const toggleH1 = document.getElementById("btnToggleH1Ema");
    if (toggleH1) {
        toggleH1.addEventListener("click", () => {
            showH1Ema = !showH1Ema;
            toggleH1.classList.toggle("active", showH1Ema);
            if (h1EmaSeries) {
                h1EmaSeries.applyOptions({ visible: showH1Ema });
            }
        });
    }
}

// ─── 5. Positions / Closed Deals Tabs ─────────────────────────────────────────
function initPositionsTabs() {
    const tabOpen = document.getElementById("tabOpenPositions");
    const tabDeals = document.getElementById("tabClosedDeals");
    const viewOpen = document.getElementById("openPositionsView");
    const viewDeals = document.getElementById("closedDealsView");

    if (tabOpen && tabDeals) {
        tabOpen.addEventListener("click", () => {
            activeTab = "open";
            tabOpen.classList.add("active");
            tabDeals.classList.remove("active");
            if (viewOpen) viewOpen.style.display = "block";
            if (viewDeals) viewDeals.style.display = "none";
        });

        tabDeals.addEventListener("click", () => {
            activeTab = "deals";
            tabDeals.classList.add("active");
            tabOpen.classList.remove("active");
            if (viewOpen) viewOpen.style.display = "none";
            if (viewDeals) viewDeals.style.display = "block";
        });
    }
}

// ─── 6. Modular Card Controls (Collapse, Maximize, Hide) ─────────────────────
function initCardControls() {
    document.querySelectorAll(".btn-card-action").forEach(btn => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            const targetId = btn.dataset.target;
            const card = document.getElementById(targetId);
            if (!card) return;

            if (btn.classList.contains("btn-collapse")) {
                card.classList.toggle("collapsed");
                btn.innerText = card.classList.contains("collapsed") ? "▴" : "▾";
                saveLayoutState();
            } else if (btn.classList.contains("btn-maximize")) {
                card.classList.toggle("maximized");
                btn.innerText = card.classList.contains("maximized") ? "🗗" : "⛶";
                if (targetId === "cardChart" && chartInstance) {
                    setTimeout(() => {
                        const container = document.getElementById("chartContainer");
                        if (container) {
                            chartInstance.applyOptions({
                                width: container.clientWidth,
                                height: container.clientHeight,
                            });
                            chartInstance.timeScale().fitContent();
                        }
                    }, 60);
                }
            } else if (btn.classList.contains("btn-hide")) {
                card.classList.add("hidden");
                const toggleInput = document.getElementById(`toggle_${targetId}`);
                if (toggleInput) toggleInput.checked = false;
                saveLayoutState();
            }
        });
    });
}

// ─── 7. Slide-Over Layout Customizer Drawer & Presets ─────────────────────────
function initLayoutDrawer() {
    const btnOpen = document.getElementById("btnCustomizeLayout");
    const drawerOverlay = document.getElementById("layoutDrawerOverlay");
    const btnClose = document.getElementById("btnCloseDrawer");
    const btnReset = document.getElementById("btnResetLayout");

    if (btnOpen && drawerOverlay) {
        btnOpen.addEventListener("click", () => {
            drawerOverlay.style.display = "flex";
        });
    }

    if (btnClose && drawerOverlay) {
        btnClose.addEventListener("click", () => {
            drawerOverlay.style.display = "none";
        });
    }

    if (drawerOverlay) {
        drawerOverlay.addEventListener("click", (e) => {
            if (e.target === drawerOverlay) drawerOverlay.style.display = "none";
        });
    }

    // Toggle checkboxes
    document.querySelectorAll(".vis-toggle-item input[type='checkbox']").forEach(input => {
        input.addEventListener("change", () => {
            const panelId = input.dataset.panel;
            const panel = document.getElementById(panelId);
            if (panel) {
                if (input.checked) {
                    panel.classList.remove("hidden");
                } else {
                    panel.classList.add("hidden");
                }
                saveLayoutState();
            }
        });
    });

    // Preset buttons
    document.querySelectorAll(".btn-preset").forEach(btn => {
        btn.addEventListener("click", () => {
            const presetKey = btn.dataset.preset;
            applyPreset(presetKey);
            document.querySelectorAll(".btn-preset").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            saveLayoutState(presetKey);
        });
    });

    // Reset layout
    if (btnReset) {
        btnReset.addEventListener("click", () => {
            localStorage.removeItem("finrl_layout_v2");
            applyPreset("default");
            document.querySelectorAll(".btn-preset").forEach(b => b.classList.remove("active"));
            const defBtn = document.querySelector(".btn-preset[data-preset='default']");
            if (defBtn) defBtn.classList.add("active");
        });
    }

    // Restore saved layout
    loadLayoutState();
}

function applyPreset(presetKey) {
    const config = LAYOUT_PRESETS[presetKey] || LAYOUT_PRESETS.default;
    const allPanels = ["cardChart", "cardGuardian", "cardPositions", "cardConsensus", "cardSpecialists", "cardCot"];

    allPanels.forEach(id => {
        const card = document.getElementById(id);
        const toggle = document.getElementById(`toggle_${id}`);
        if (!card) return;

        const isVis = config.visible.includes(id);
        const isCol = (config.collapsed || []).includes(id);

        card.classList.toggle("hidden", !isVis);
        card.classList.toggle("collapsed", isCol);
        if (toggle) toggle.checked = isVis;

        // Reset maximize state
        card.classList.remove("maximized");
    });

    if (chartInstance) {
        setTimeout(() => {
            const container = document.getElementById("chartContainer");
            if (container) {
                chartInstance.applyOptions({
                    width: container.clientWidth,
                    height: container.clientHeight,
                });
                chartInstance.timeScale().fitContent();
            }
        }, 60);
    }
}

function saveLayoutState(activePreset = null) {
    const allPanels = ["cardChart", "cardGuardian", "cardPositions", "cardConsensus", "cardSpecialists", "cardCot"];
    const visible = [];
    const collapsed = [];

    allPanels.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            if (!el.classList.contains("hidden")) visible.push(id);
            if (el.classList.contains("collapsed")) collapsed.push(id);
        }
    });

    const stateObj = {
        preset: activePreset,
        visible,
        collapsed
    };

    localStorage.setItem("finrl_layout_v2", JSON.stringify(stateObj));
}

function loadLayoutState() {
    const raw = localStorage.getItem("finrl_layout_v2");
    if (!raw) return;

    try {
        const saved = JSON.parse(raw);
        if (saved.preset && LAYOUT_PRESETS[saved.preset]) {
            applyPreset(saved.preset);
            document.querySelectorAll(".btn-preset").forEach(b => {
                b.classList.toggle("active", b.dataset.preset === saved.preset);
            });
            return;
        }

        const allPanels = ["cardChart", "cardGuardian", "cardPositions", "cardConsensus", "cardSpecialists", "cardCot"];
        allPanels.forEach(id => {
            const card = document.getElementById(id);
            const toggle = document.getElementById(`toggle_${id}`);
            if (!card) return;

            const isVis = saved.visible ? saved.visible.includes(id) : true;
            const isCol = saved.collapsed ? saved.collapsed.includes(id) : false;

            card.classList.toggle("hidden", !isVis);
            card.classList.toggle("collapsed", isCol);
            if (toggle) toggle.checked = isVis;
        });
    } catch (e) {
        console.warn("Layout restore error:", e);
    }
}

// ─── 8. Event Listeners ─────────────────────────────────────────────────────
function setupEventListeners() {
    const sel = document.getElementById("symbolSelect");
    if (sel) {
        sel.addEventListener("change", (e) => {
            activeSymbol = e.target.value;
            fetchChartBars(activeSymbol, activeTimeframe);
            fetchCouncilDecision(activeSymbol);
        });
    }

    const btnDelib = document.getElementById("btnDeliberate");
    if (btnDelib) {
        btnDelib.addEventListener("click", () => {
            fetchCouncilDecision(activeSymbol, true); // Force recalculation with live models
        });
    }

    const btnRef = document.getElementById("btnRefreshChart");
    if (btnRef) {
        btnRef.addEventListener("click", () => {
            fetchChartBars(activeSymbol, activeTimeframe);
            fetchAccount();
        });
    }
}

// ─── 9. Theme Management (Dark / White Mode) ─────────────────────────────────
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

// ─── 10. Support & Donation Modal ─────────────────────────────────────────────
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
                        copyBtn.style.background = "";
                    }, 2000);
                }
            });
        });
    }
}
