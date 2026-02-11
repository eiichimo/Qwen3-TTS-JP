# coding=utf-8
"""Custom CSS theme and JavaScript for Qwen3-TTS UI."""

CUSTOM_CSS = """
/* ===== Root Variables ===== */
:root {
    --primary-color: #4A90D9;
    --primary-hover: #5BA0E9;
    --secondary-color: #6C5CE7;
    --accent-color: #00CEC9;
    --bg-dark: #1a1a2e;
    --bg-card: #16213e;
    --bg-surface: #0f3460;
    --text-primary: #e8e8e8;
    --text-secondary: #a0a0b0;
    --border-color: #2a2a4a;
    --success-color: #00b894;
    --warning-color: #fdcb6e;
    --error-color: #e17055;
    --gradient-header: linear-gradient(135deg, #0f3460 0%, #1a1a2e 50%, #16213e 100%);
    --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.1);
    --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.15);
    --shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.2);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
}

/* ===== Global ===== */
.gradio-container {
    max-width: none !important;
}

/* ===== Header ===== */
.app-header {
    background: var(--gradient-header);
    border-radius: var(--radius-lg);
    padding: 28px 36px 20px;
    margin-bottom: 16px;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-lg);
    position: relative;
    overflow: hidden;
}

.app-header::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(74,144,217,0.15) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(0,206,201,0.10) 0%, transparent 60%);
    pointer-events: none;
}

.app-header h1 {
    margin: 0 0 6px 0;
    font-size: 2.2em;
    font-weight: 700;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    letter-spacing: 0.02em;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    position: relative;
}

.app-header p {
    margin: 4px 0;
    color: var(--text-secondary);
    font-size: 0.9em;
    position: relative;
}

/* ===== Primary Button Animation ===== */
.primary-btn {
    transition: all 0.3s ease !important;
    box-shadow: var(--shadow-sm) !important;
}
.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-lg) !important;
}
.primary-btn:active {
    transform: translateY(0px) !important;
}

/* ===== Status Dots ===== */
.status-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.status-dot.green { background-color: var(--success-color); }
.status-dot.yellow { background-color: var(--warning-color); }
.status-dot.red { background-color: var(--error-color); }

/* ===== VRAM Progress Bar ===== */
.vram-bar-container {
    background: var(--bg-card);
    border-radius: var(--radius-sm);
    overflow: hidden;
    height: 24px;
    margin: 8px 0;
    border: 1px solid var(--border-color);
}
.vram-bar {
    height: 100%;
    border-radius: var(--radius-sm);
    transition: width 0.5s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8em;
    font-weight: 600;
    color: white;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}
.vram-bar.low { background: linear-gradient(90deg, var(--success-color), #55efc4); }
.vram-bar.medium { background: linear-gradient(90deg, var(--warning-color), #ffeaa7); }
.vram-bar.high { background: linear-gradient(90deg, var(--error-color), #fab1a0); }

/* ===== Preset Cards ===== */
.preset-card {
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    transition: all 0.2s ease !important;
    min-height: auto !important;
}
.preset-card:hover {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 1px var(--primary-color) !important;
}

/* ===== Tab Styling ===== */
.tab-nav button {
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}

/* ===== Info Cards ===== */
.info-card {
    border-radius: var(--radius-md);
    padding: 16px;
    margin: 8px 0;
    border: 1px solid var(--border-color);
}

/* ===== Disclaimer ===== */
.disclaimer {
    opacity: 0.7;
    font-size: 0.85em;
    border-top: 1px solid var(--border-color);
    padding-top: 16px;
    margin-top: 24px;
}

/* ===== Language Selector (inside header) ===== */
.lang-selector {
    max-width: 200px;
    margin-left: auto;
}
.app-header .lang-selector select,
.app-header .lang-selector input {
    background: rgba(255,255,255,0.1) !important;
    border-color: rgba(255,255,255,0.2) !important;
    color: var(--text-primary) !important;
}
.app-header .lang-selector {
    padding-top: 8px;
}

/* Remove ALL grey backgrounds inside header */
.app-header div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
/* Keep the dropdown input itself slightly visible */
.app-header .lang-selector input,
.app-header .lang-selector select,
.app-header .lang-selector [data-testid="textbox"],
.app-header .lang-selector ul {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: var(--radius-sm) !important;
}

/* ===== Scrollbar Hidden ===== */
::-webkit-scrollbar {
    width: 0px;
    background: transparent;
}

/* ===== Tips Section ===== */
.tips-section {
    border-left: 3px solid var(--primary-color);
    padding-left: 12px;
    margin: 12px 0;
    opacity: 0.9;
}

/* ===== Section Title ===== */
.section-title {
    font-size: 1.1em;
    font-weight: 600;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid var(--primary-color);
    display: inline-block;
}
"""

UPLOAD_FIX_JS = """
() => {
    // Ensure Gradio upload area labels stay consistent
    const observer = new MutationObserver(() => {
        document.querySelectorAll('.upload-text').forEach(el => {
            const spans = el.querySelectorAll('span');
            spans.forEach(span => {
                if (span.textContent && /ドラッグ|ドロップ/.test(span.textContent)) {
                    span.textContent = span.textContent
                        .replace(/ドラッグ.*ドロップ/g, 'Drag & Drop');
                }
            });
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
}
"""
