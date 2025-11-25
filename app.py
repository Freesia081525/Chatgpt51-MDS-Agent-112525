import os
import io
import base64
import json
from typing import Dict, Any, List, Optional

import streamlit as st
import yaml
import plotly.graph_objects as go

# --- LLM SDKs ---
from google import genai  # pip install google-genai
from openai import OpenAI  # pip install openai
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user as grok_user, system as grok_system

# For basic PDF text extraction (not OCR)
import PyPDF2  # pip install PyPDF2

# ===============================
# 0. BASIC CONFIG
# ===============================
st.set_page_config(
    page_title="Floral Agentic Workflow",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===============================
# 1. LOCALIZATION
# ===============================
def t(key: str, lang: str) -> str:
    """Simple localization lookup."""
    TEXT = {
        "app_title": {
            "en": "Floral Agentic Workflow Dashboard",
            "zh": "花語智能代理工作台",
        },
        "upload_label": {
            "en": "Upload a document (PDF, TXT, MD, JSON) or paste text",
            "zh": "上傳文件（PDF/TXT/MD/JSON）或貼上文字",
        },
        "paste_text": {"en": "Or paste text below:", "zh": "或在下方貼上文字："},
        "global_task": {"en": "Global task / question", "zh": "全域任務 / 問題說明"},
        "run_agents": {"en": "Run Selected Agents", "zh": "執行選取的代理"},
        "settings": {"en": "Settings", "zh": "設定"},
        "api_keys": {"en": "API Keys", "zh": "API 金鑰"},
        "theme_settings": {"en": "Theme & Language", "zh": "主題與語言"},
        "dashboard": {"en": "Agent Dashboard", "zh": "代理儀表板"},
        "agents_panel": {"en": "Agents", "zh": "代理清單"},
        "prompt_label": {"en": "Advanced global system prompt", "zh": "進階全域系統提示詞"},
        "max_tokens": {"en": "Max tokens", "zh": "最大 Token 數"},
        "model": {"en": "Model", "zh": "模型"},
        "status": {"en": "Status", "zh": "狀態"},
        "output": {"en": "Output", "zh": "輸出"},
        "no_doc": {"en": "No document or text provided yet.", "zh": "尚未提供任何文件或文字。"},
        "selected_theme": {"en": "Selected Flower Theme", "zh": "目前花卉主題"},
        "spin_wheel": {"en": "Spin Floral Luck Wheel", "zh": "旋轉花語幸運輪"},
        "light_mode": {"en": "Light mode", "zh": "亮色模式"},
        "dark_mode": {"en": "Dark mode", "zh": "深色模式"},
        "language": {"en": "Language", "zh": "介面語言"},
        "english": {"en": "English", "zh": "英文"},
        "traditional_chinese": {"en": "Traditional Chinese", "zh": "繁體中文"},
        "edit_agent_prompt": {"en": "Edit agent prompt", "zh": "編輯代理提示詞"},
        "wow_status": {"en": "WOW Status Indicators", "zh": "WOW 狀態指示"},
        "document_preview": {"en": "Document Preview", "zh": "文件預覽"},
        "token_usage": {"en": "Token Usage (approx.)", "zh": "Token 使用量（約略）"},
        "response_length": {"en": "Response length (chars)", "zh": "回應長度（字元）"},
    }
    if key not in TEXT:
        return key
    return TEXT[key]["zh"] if lang == "zh" else TEXT[key]["en"]


# ===============================
# 2. THEMES: 20 FLOWER STYLES
# ===============================
FLOWER_THEMES = [
    {
        "id": "sakura_breeze",
        "label": "Sakura Breeze",
        "emoji": "🌸",
        "light": {"bg": "#fff5f8", "fg": "#3b0b19", "accent": "#ff99c8"},
        "dark": {"bg": "#2b0f1b", "fg": "#ffe6f2", "accent": "#ff7aa2"},
    },
    {
        "id": "rose_gold",
        "label": "Rose Gold",
        "emoji": "🌹",
        "light": {"bg": "#fff6f7", "fg": "#4b1114", "accent": "#f75c77"},
        "dark": {"bg": "#2a0d0f", "fg": "#ffe8ec", "accent": "#ff6b81"},
    },
    {
        "id": "lavender_dream",
        "label": "Lavender Dream",
        "emoji": "💜",
        "light": {"bg": "#f4f1ff", "fg": "#22164d", "accent": "#a78bfa"},
        "dark": {"bg": "#1b1433", "fg": "#ede9fe", "accent": "#c4b5fd"},
    },
    {
        "id": "sunflower_glow",
        "label": "Sunflower Glow",
        "emoji": "🌻",
        "light": {"bg": "#fffbea", "fg": "#3b2f0c", "accent": "#fbbf24"},
        "dark": {"bg": "#1f1303", "fg": "#fef3c7", "accent": "#facc15"},
    },
    {
        "id": "lotus_pond",
        "label": "Lotus Pond",
        "emoji": "🪷",
        "light": {"bg": "#ecfdf5", "fg": "#064e3b", "accent": "#22c55e"},
        "dark": {"bg": "#022c22", "fg": "#dcfce7", "accent": "#4ade80"},
    },
    {
        "id": "orchid_mist",
        "label": "Orchid Mist",
        "emoji": "🌺",
        "light": {"bg": "#fdf2ff", "fg": "#3b0764", "accent": "#e879f9"},
        "dark": {"bg": "#2b0b39", "fg": "#fae8ff", "accent": "#f472b6"},
    },
    {
        "id": "peony_blush",
        "label": "Peony Blush",
        "emoji": "🌷",
        "light": {"bg": "#fff1f2", "fg": "#4a041c", "accent": "#fb7185"},
        "dark": {"bg": "#3f0213", "fg": "#ffe4e6", "accent": "#fb7185"},
    },
    {
        "id": "iris_night",
        "label": "Iris Night",
        "emoji": "🪻",
        "light": {"bg": "#eff6ff", "fg": "#111827", "accent": "#6366f1"},
        "dark": {"bg": "#020617", "fg": "#e5e7eb", "accent": "#4f46e5"},
    },
    {
        "id": "cherry_meadow",
        "label": "Cherry Meadow",
        "emoji": "🍒",
        "light": {"bg": "#fef2f2", "fg": "#111827", "accent": "#fb7185"},
        "dark": {"bg": "#111827", "fg": "#f9fafb", "accent": "#f97316"},
    },
    {
        "id": "camellia_silk",
        "label": "Camellia Silk",
        "emoji": "🌺",
        "light": {"bg": "#fdf2f8", "fg": "#4a044e", "accent": "#ec4899"},
        "dark": {"bg": "#3b0764", "fg": "#fdf2f8", "accent": "#db2777"},
    },
    {
        "id": "magnolia_cloud",
        "label": "Magnolia Cloud",
        "emoji": "🌼",
        "light": {"bg": "#f9fafb", "fg": "#111827", "accent": "#eab308"},
        "dark": {"bg": "#0b1120", "fg": "#e5e7eb", "accent": "#f59e0b"},
    },
    {
        "id": "plum_blossom",
        "label": "Plum Blossom",
        "emoji": "🌸",
        "light": {"bg": "#fef2ff", "fg": "#4a044e", "accent": "#f97316"},
        "dark": {"bg": "#3f0e40", "fg": "#fce7f3", "accent": "#f97316"},
    },
    {
        "id": "gardenia_moon",
        "label": "Gardenia Moon",
        "emoji": "🌙",
        "light": {"bg": "#f9fafb", "fg": "#020617", "accent": "#22c55e"},
        "dark": {"bg": "#020617", "fg": "#e5e7eb", "accent": "#22c55e"},
    },
    {
        "id": "wisteria_rain",
        "label": "Wisteria Rain",
        "emoji": "🌧️",
        "light": {"bg": "#eef2ff", "fg": "#1e293b", "accent": "#a855f7"},
        "dark": {"bg": "#020617", "fg": "#e5e7eb", "accent": "#8b5cf6"},
    },
    {
        "id": "dahlia_fire",
        "label": "Dahlia Fire",
        "emoji": "🔥",
        "light": {"bg": "#fff7ed", "fg": "#1f2937", "accent": "#f97316"},
        "dark": {"bg": "#111827", "fg": "#f9fafb", "accent": "#fb923c"},
    },
    {
        "id": "bluebell_forest",
        "label": "Bluebell Forest",
        "emoji": "🔵",
        "light": {"bg": "#eff6ff", "fg": "#111827", "accent": "#3b82f6"},
        "dark": {"bg": "#020617", "fg": "#e5e7eb", "accent": "#60a5fa"},
    },
    {
        "id": "poppy_fields",
        "label": "Poppy Fields",
        "emoji": "🌺",
        "light": {"bg": "#fef2f2", "fg": "#1f2937", "accent": "#ef4444"},
        "dark": {"bg": "#111827", "fg": "#f9fafb", "accent": "#f87171"},
    },
    {
        "id": "lotus_dawn",
        "label": "Lotus Dawn",
        "emoji": "🌅",
        "light": {"bg": "#fefce8", "fg": "#1f2937", "accent": "#22c55e"},
        "dark": {"bg": "#1e293b", "fg": "#e5e7eb", "accent": "#10b981"},
    },
    {
        "id": "hibiscus_sunset",
        "label": "Hibiscus Sunset",
        "emoji": "🌇",
        "light": {"bg": "#fff7ed", "fg": "#1f2937", "accent": "#fb7185"},
        "dark": {"bg": "#1f2937", "fg": "#e5e7eb", "accent": "#f97316"},
    },
    {
        "id": "jasmine_night",
        "label": "Jasmine Night",
        "emoji": "🌙",
        "light": {"bg": "#f9fafb", "fg": "#1f2937", "accent": "#22c55e"},
        "dark": {"bg": "#020617", "fg": "#e5e7eb", "accent": "#84cc16"},
    },
]


def apply_theme(theme_id: str, dark: bool):
    theme = next((t for t in FLOWER_THEMES if t["id"] == theme_id), FLOWER_THEMES[0])
    palette = theme["dark"] if dark else theme["light"]

    css = f"""
    <style>
    body {{
        background: {palette['bg']} !important;
        color: {palette['fg']} !important;
    }}
    .stApp {{
        background: linear-gradient(135deg, {palette['bg']} 0%, #ffffff11 50%, {palette['bg']} 100%);
    }}
    .stMarkdown, .stTextInput, .stTextArea, .stSelectbox, .stDataFrame, .stButton > button {{
        color: {palette['fg']} !important;
    }}
    .floral-accent {{
        border-radius: 999px;
        padding: 0.4rem 0.9rem;
        background: {palette['accent']}22;
        border: 1px solid {palette['accent']};
        color: {palette['fg']};
        font-weight: 600;
    }}
    .floral-badge-success {{
        background: #16a34a22;
        border-color: #22c55e;
    }}
    .floral-badge-error {{
        background: #b91c1c22;
        border-color: #ef4444;
    }}
    .floral-badge-running {{
        background: #0369a122;
        border-color: #0ea5e9;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    return theme


# ===============================
# 3. LOAD AGENTS
# ===============================
@st.cache_resource
def load_agents_config(path: str = "agents.yaml") -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("agents", [])


AGENTS_BASE = load_agents_config()

# Available models for quick selection
AVAILABLE_MODELS = [
    # Gemini
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    # OPENAAI/OpenAI-style
    "gpt-5-nano",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    # Grok
    "grok-4-fast-reasoning",
    "grok-3-mini",
]


# ===============================
# 4. SESSION STATE INIT
# ===============================
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "theme_id" not in st.session_state:
    st.session_state.theme_id = FLOWER_THEMES[0]["id"]
if "agents" not in st.session_state:
    # Make a mutable copy of base config
    st.session_state.agents = [
        {
            **agent,
            "status": "Pending",
            "output": "",
            "token_usage": 0,
        }
        for agent in AGENTS_BASE
    ]
if "global_prompt" not in st.session_state:
    st.session_state.global_prompt = (
        "You are part of a multi-agent analysis system called 'Floral Agentic Workflow'."
    )
if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {
        "gemini": os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", ""),
        "openai": os.getenv("OPENAI_API_KEY", "") or os.getenv("OPENAAI_API_KEY", ""),
        "grok": os.getenv("XAI_API_KEY", ""),
    }

# Apply theme
current_theme = apply_theme(st.session_state.theme_id, st.session_state.dark_mode)

# ===============================
# 5. SIDEBAR: SETTINGS, THEME, API KEYS
# ===============================
with st.sidebar:
    st.markdown(f"### {t('settings', st.session_state.lang)}")

    # Language toggle
    lang_choice = st.radio(
        t("language", st.session_state.lang),
        options=["en", "zh"],
        format_func=lambda x: t(
            "english" if x == "en" else "traditional_chinese", st.session_state.lang
        ),
        index=0 if st.session_state.lang == "en" else 1,
    )
    st.session_state.lang = lang_choice

    # Light/Dark
    mode = st.radio(
        t("light_mode", st.session_state.lang) + " / " + t("dark_mode", st.session_state.lang),
        options=["light", "dark"],
        index=1 if st.session_state.dark_mode else 0,
    )
    st.session_state.dark_mode = mode == "dark"
    current_theme = apply_theme(st.session_state.theme_id, st.session_state.dark_mode)

    # Flower Luck Wheel
    st.markdown(f"### {t('theme_settings', st.session_state.lang)}")
    theme_labels = [f"{th['emoji']} {th['label']}" for th in FLOWER_THEMES]
    # Simple "wheel" using plotly pie
    wheel_fig = go.Figure(
        data=[
            go.Pie(
                labels=theme_labels,
                values=[1] * len(FLOWER_THEMES),
                hole=0.4,
                textinfo="none",
            )
        ]
    )
    wheel_fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=220,
    )
    st.plotly_chart(wheel_fig, use_container_width=True)

    if st.button(t("spin_wheel", st.session_state.lang)):
        import random

        st.session_state.theme_id = random.choice(FLOWER_THEMES)["id"]
        current_theme = apply_theme(st.session_state.theme_id, st.session_state.dark_mode)

    current_theme_obj = next(
        (th for th in FLOWER_THEMES if th["id"] == st.session_state.theme_id), FLOWER_THEMES[0]
    )
    st.markdown(
        f"**{t('selected_theme', st.session_state.lang)}:** "
        f"{current_theme_obj['emoji']} {current_theme_obj['label']}"
    )

    # API KEYS
    st.markdown(f"### {t('api_keys', st.session_state.lang)}")

    # We do NOT display env keys; only show if empty for input.
    if not st.session_state.api_keys["gemini"]:
        gemini_key = st.text_input("Gemini API Key", type="password")
        if gemini_key:
            st.session_state.api_keys["gemini"] = gemini_key

    if not st.session_state.api_keys["openai"]:
        openai_key = st.text_input("OPENAAI/OpenAI API Key", type="password")
        if openai_key:
            st.session_state.api_keys["openai"] = openai_key

    if not st.session_state.api_keys["grok"]:
        grok_key = st.text_input("Grok XAI_API_KEY", type="password")
        if grok_key:
            st.session_state.api_keys["grok"] = grok_key

    st.markdown(
        "<p style='font-size: 0.75rem; opacity:0.8;'>"
        "API keys are kept in session memory only and sent directly to the provider APIs."
        "</p>",
        unsafe_allow_html=True,
    )

# ===============================
# 6. LLM CLIENT HELPERS
# ===============================
def get_gemini_client():
    key = st.session_state.api_keys.get("gemini", "")
    if not key:
        raise RuntimeError("Missing Gemini API key")
    return genai.Client(api_key=key)


def get_openai_client():
    key = st.session_state.api_keys.get("openai", "")
    if not key:
        raise RuntimeError("Missing OPENAAI/OpenAI API key")
    return OpenAI(api_key=key)


def get_grok_client():
    key = st.session_state.api_keys.get("grok", "")
    if not key:
        raise RuntimeError("Missing Grok XAI_API_KEY")
    return XAIClient(api_key=key, timeout=3600)


def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    """
    Unified LLM call across Gemini, OPENAAI/OpenAI, Grok.
    Returns {text, usage_approx}
    """
    # Decide provider
    lower = model.lower()
    if lower.startswith("gemini"):
        client = get_gemini_client()
        prompt = f"{system_prompt}\n\n{user_prompt}"
        resp = client.models.generate_content(
            model=model,
            contents=[{"role": "user", "parts": [prompt]}],
            config={"max_output_tokens": max_tokens},
        )
        text = getattr(resp, "text", "") or ""
        # usage is not uniform; approximate by length
        return {"text": text, "usage_approx": len(prompt) + len(text)}

    elif lower.startswith("gpt-"):
        client = get_openai_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content
        usage_obj = getattr(resp, "usage", None)
        if usage_obj:
            usage_approx = usage_obj.total_tokens
        else:
            usage_approx = len(system_prompt) + len(user_prompt) + len(text)
        return {"text": text, "usage_approx": usage_approx}

    elif lower.startswith("grok-"):
        client = get_grok_client()
        chat = client.chat.create(model=model)
        chat.append(grok_system(system_prompt))
        chat.append(grok_user(user_prompt))
        resp = chat.sample()
        # Sample code returns resp.content; we cast to str for safety.
        text = str(resp.content)
        usage_approx = len(system_prompt) + len(user_prompt) + len(text)
        return {"text": text, "usage_approx": usage_approx}

    else:
        # Fallback error
        raise ValueError(f"Unsupported model: {model}")


# ===============================
# 7. DOCUMENT INGESTION
# ===============================
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PyPDF2.PdfReader(uploaded_file)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n\n".join(text)


# ===============================
# 8. MAIN LAYOUT
# ===============================
st.markdown(f"## {t('app_title', st.session_state.lang)}")

# Top columns: Document + Global Prompt
col_doc, col_prompt = st.columns([1.3, 1])

with col_doc:
    st.markdown(f"#### {t('upload_label', st.session_state.lang)}")
    uploaded = st.file_uploader(
        "",
        type=["pdf", "txt", "md", "markdown", "json"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        suffix = uploaded.name.lower().split(".")[-1]
        if suffix == "pdf":
            st.session_state.doc_text = extract_text_from_pdf(uploaded)
        elif suffix in ["txt", "md", "markdown"]:
            st.session_state.doc_text = uploaded.read().decode("utf-8", errors="ignore")
        elif suffix == "json":
            raw = uploaded.read().decode("utf-8", errors="ignore")
            try:
                data = json.loads(raw)
                st.session_state.doc_text = json.dumps(data, indent=2, ensure_ascii=False)
            except Exception:
                st.session_state.doc_text = raw

    st.markdown(f"**{t('document_preview', st.session_state.lang)}**")
    if st.session_state.doc_text:
        st.text_area(
            "",
            value=st.session_state.doc_text[:8000],
            height=220,
            key="doc_preview",
        )
    else:
        st.info(t("no_doc", st.session_state.lang))

    st.markdown(f"**{t('paste_text', st.session_state.lang)}**")
    pasted = st.text_area(
        "",
        value="",
        height=160,
        key="doc_paste",
        placeholder="Type or paste additional content here...",
    )
    if pasted:
        # Merge pasted with existing
        st.session_state.doc_text = (st.session_state.doc_text + "\n\n" + pasted).strip()

with col_prompt:
    st.markdown(f"#### {t('global_task', st.session_state.lang)}")
    global_task = st.text_area(
        "",
        height=140,
        key="global_task",
        placeholder="e.g., 'Summarize this document for executives and list open risks.'",
    )
    st.markdown(f"#### {t('prompt_label', st.session_state.lang)}")
    st.session_state.global_prompt = st.text_area(
        "",
        value=st.session_state.global_prompt,
        height=220,
        key="advanced_prompt",
    )

# ===============================
# 9. AGENT CONFIG PANEL
# ===============================
st.markdown("---")
st.markdown(f"### {t('agents_panel', st.session_state.lang)}")

if not st.session_state.agents:
    st.warning("No agents found in agents.yaml")
else:
    # Editable table-like UI
    for i, agent in enumerate(st.session_state.agents):
        with st.expander(f"{agent.get('name', agent['id'])} [{agent['id']}]"):
            c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
            with c1:
                st.checkbox(
                    "Enabled",
                    value=agent.get("enabled", True),
                    key=f"agent_enabled_{agent['id']}",
                )
            with c2:
                st.session_state.agents[i]["model"] = st.selectbox(
                    t("model", st.session_state.lang),
                    options=AVAILABLE_MODELS,
                    index=AVAILABLE_MODELS.index(agent["model"])
                    if agent["model"] in AVAILABLE_MODELS
                    else 0,
                    key=f"agent_model_{agent['id']}",
                )
            with c3:
                st.session_state.agents[i]["max_tokens"] = st.slider(
                    t("max_tokens", st.session_state.lang),
                    min_value=100,
                    max_value=12000,
                    value=int(agent.get("max_tokens", 2048)),
                    step=100,
                    key=f"agent_maxtok_{agent['id']}",
                )
            with c4:
                st.markdown(
                    f"<span class='floral-accent'>{t('status', st.session_state.lang)}: "
                    f"{agent.get('status', 'Pending')}</span>",
                    unsafe_allow_html=True,
                )

            if st.button(t("edit_agent_prompt", st.session_state.lang), key=f"edit_prompt_{agent['id']}"):
                st.session_state[f"show_prompt_modal_{agent['id']}"] = True

            # Modal-style prompt editor
            if st.session_state.get(f"show_prompt_modal_{agent['id']}", False):
                st.markdown("---")
                st.markdown(f"**System Prompt for {agent['name']}**")
                new_prompt = st.text_area(
                    "",
                    value=agent.get("system_prompt", ""),
                    height=220,
                    key=f"prompt_text_{agent['id']}",
                )
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("Save", key=f"save_prompt_{agent['id']}"):
                        st.session_state.agents[i]["system_prompt"] = new_prompt
                        st.session_state[f"show_prompt_modal_{agent['id']}"] = False
                with col_cancel:
                    if st.button("Cancel", key=f"cancel_prompt_{agent['id']}"):
                        st.session_state[f"show_prompt_modal_{agent['id']}"] = False

# ===============================
# 10. RUN AGENTS
# ===============================
st.markdown("---")
if st.button(t("run_agents", st.session_state.lang)):
    if not st.session_state.doc_text and not global_task:
        st.warning("Please provide document text or a task before running agents.")
    else:
        for i, agent in enumerate(st.session_state.agents):
            enabled = st.session_state.get(f"agent_enabled_{agent['id']}", agent.get("enabled", True))
            if not enabled:
                st.session_state.agents[i]["status"] = "Skipped"
                continue

            st.session_state.agents[i]["status"] = "Running"
            with st.spinner(f"Running agent: {agent['name']} ({agent['model']})"):
                try:
                    # Build user prompt
                    composed_user_prompt = ""
                    if global_task:
                        composed_user_prompt += f"Global task:\n{global_task}\n\n"
                    if st.session_state.doc_text:
                        composed_user_prompt += "Document content:\n" + st.session_state.doc_text[:12000]
                    else:
                        composed_user_prompt += "(No document provided.)"

                    full_system_prompt = (
                        st.session_state.global_prompt.strip()
                        + "\n\n--- Agent-specific instructions ---\n"
                        + agent.get("system_prompt", "").strip()
                    )

                    result = call_llm(
                        model=agent["model"],
                        system_prompt=full_system_prompt,
                        user_prompt=composed_user_prompt,
                        max_tokens=int(agent.get("max_tokens", 2048)),
                    )
                    st.session_state.agents[i]["output"] = result["text"]
                    st.session_state.agents[i]["token_usage"] = result["usage_approx"]
                    st.session_state.agents[i]["status"] = "Success"
                except Exception as e:
                    st.session_state.agents[i]["output"] = f"Error: {e}"
                    st.session_state.agents[i]["token_usage"] = 0
                    st.session_state.agents[i]["status"] = "Error"


# ===============================
# 11. WOW STATUS INDICATORS + DASHBOARD
# ===============================
st.markdown("---")
st.markdown(f"### {t('wow_status', st.session_state.lang)} & {t('dashboard', st.session_state.lang)}")

if st.session_state.agents:
    # Status badges
    cols = st.columns(len(st.session_state.agents))
    for col, agent in zip(cols, st.session_state.agents):
        status = agent.get("status", "Pending")
        if status == "Success":
            badge_class = "floral-badge-success"
        elif status == "Error":
            badge_class = "floral-badge-error"
        elif status == "Running":
            badge_class = "floral-badge-running"
        else:
            badge_class = ""
        with col:
            st.markdown(
                f"<div class='floral-accent {badge_class}'>"
                f"{agent.get('name', agent['id'])}<br/>"
                f"<small>Status: {status}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Interactive dashboard
    names = [a.get("name", a["id"]) for a in st.session_state.agents]
    token_usage = [a.get("token_usage", 0) for a in st.session_state.agents]
    resp_lengths = [len(a.get("output", "") or "") for a in st.session_state.agents]

    dash_col1, dash_col2 = st.columns(2)
    with dash_col1:
        st.markdown(f"**{t('token_usage', st.session_state.lang)}**")
        fig_tokens = go.Figure(
            data=[go.Bar(x=names, y=token_usage, marker_color=current_theme["light"]["accent"])]
        )
        fig_tokens.update_layout(
            xaxis_title="Agent",
            yaxis_title="Tokens (approx.)",
            height=320,
            margin=dict(l=40, r=20, t=30, b=80),
        )
        st.plotly_chart(fig_tokens, use_container_width=True)

    with dash_col2:
        st.markdown(f"**{t('response_length', st.session_state.lang)}**")
        fig_len = go.Figure(
            data=[go.Bar(x=names, y=resp_lengths, marker_color=current_theme["light"]["accent"])]
        )
        fig_len.update_layout(
            xaxis_title="Agent",
            yaxis_title="Characters",
            height=320,
            margin=dict(l=40, r=20, t=30, b=80),
        )
        st.plotly_chart(fig_len, use_container_width=True)


# ===============================
# 12. PER-AGENT OUTPUT VIEW
# ===============================
st.markdown("---")
st.markdown(f"### {t('output', st.session_state.lang)}")

for agent in st.session_state.agents:
    with st.expander(f"{agent.get('name', agent['id'])} ({agent['status']})"):
        st.markdown(f"**Model:** `{agent['model']}`  |  **Tokens (approx):** {agent.get('token_usage', 0)}")
        st.markdown(agent.get("output", "") or "_No output yet._")
