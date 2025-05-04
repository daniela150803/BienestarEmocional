import os
import json
import hashlib
import requests
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import io
import re
from collections import Counter
from transformers import pipeline

# ---------------------------------------------
# 1. Modelo de sentimiento (BERT)
# ---------------------------------------------
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device="cpu"
    )
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    sentiment_pipeline = None

# ---------------------------------------------
# 2. Reglas 
# ---------------------------------------------

def fetch_rules():
    with open("rules.json", "r", encoding="utf-8") as f:
        return json.load(f)

def eval_rules(estado):
    recomendaciones = []
    reglas = fetch_rules()
    for regla in reglas:
        cond = regla.get("condition", {})
        cumple = True
        for var, expr in cond.items():
            val = estado.get(var, None)
            if isinstance(expr, str) and expr[0] in "<>":
                try:
                    op, umbral = expr[0], float(expr[1:])
                    val = float(val) if val is not None else 0
                    if op == "<" and not (val < umbral): cumple = False
                    if op == ">" and not (val > umbral): cumple = False
                except:
                    cumple = False
            else:
                if str(val).strip().lower() != str(expr).strip().lower():
                    cumple = False
            if not cumple:
                break
        if cumple:
            recomendaciones.append({
                "advice": regla.get("advice", ""),
                "source": regla.get("source", "")
            })
    return recomendaciones


# ---------------------------------------------
# 3. Perfil de usuario
# ---------------------------------------------
PROFILE_PATH = "user_profiles.json"

def load_profiles():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r") as f:
            return json.load(f)
    return {}

def save_profiles(profiles):
    with open(PROFILE_PATH, "w") as f:
        json.dump(profiles, f, indent=2)

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ---------------------------------------------
# 4. Clasificación + generación de consejos
# ---------------------------------------------
def clasificar_y_recomendar(full_text: str, latest_entry: str):
    if not sentiment_pipeline:
        return [], "Servicio no disponible"
    res = sentiment_pipeline(full_text[:512])[0]
    score = int(res["label"].split()[0])
    category = ("Necesita apoyo", "Neutral", "Bienestar alto")[min(score-1, 2)]

    parts = {}
    for p in latest_entry.split(";"):
        if ":" in p:
            k, v = p.split(":", 1)
            parts[k] = v
    try:
        sleep_hours = float(parts.get("Sueño", "0").split()[0])
    except:
        sleep_hours = 0.0
    try:
        stress_level = int(parts.get("Estres", "0"))
    except:
        stress_level = 0
    emotion = parts.get("Emoción", "").strip()
    estado = {
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "emotion": emotion
    }
    recs = eval_rules(estado)
    header = f"💬 *Puntuación* {score}/5 → **{category}**\n"
    advice_lines = "\n".join(f"- {r['advice']} *(fuente: {r['source']})*" for r in recs)
    if not advice_lines:
        advice_lines = "Por ahora no hay consejos nuevos, ¡hablemos mañana!"
    return recs, header + advice_lines

# ---------------------------------------------
# 5. Diagnóstico visual
# ---------------------------------------------
def extraer_num(texto):
    match = re.search(r"(\d+\.?\d*)", texto)
    return float(match.group(1)) if match else 0

def generar_diagnostico_completo(uid):
    profiles = load_profiles()
    entries = profiles[uid]["entries"]
    sueño, estres, emociones, cafe, ejercicio = [], [], [], [], []
    for e in entries:
        if ":" not in e: continue
        partes = dict(p.split(":", 1) for p in e.split(";") if ":" in p)
        sueño.append(extraer_num(partes.get("Sueño", "")))
        estres.append(extraer_num(partes.get("Estres", "")))
        emociones.append(partes.get("Emoción", "Desconocida").strip())
        cafe.append(extraer_num(partes.get("Café", "")))
        ejercicio.append(extraer_num(partes.get("Ejercicio", "")))

    dias = [f"Día {i+1}" for i in range(len(sueño))]
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    axs[0, 0].plot(dias, sueño, marker='o'); axs[0, 0].set_title(" Sueño (horas)"); axs[0, 0].grid(True); axs[0, 0].tick_params(axis='x', rotation=45)
    axs[0, 1].plot(dias, estres, marker='x', color='red'); axs[0, 1].set_title(" Estrés (1–10)"); axs[0, 1].grid(True); axs[0, 1].tick_params(axis='x', rotation=45)
    axs[1, 0].bar(dias, cafe, color='brown'); axs[1, 0].set_title(" Café"); axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 1].bar(dias, ejercicio, color='green'); axs[1, 1].set_title(" Ejercicio"); axs[1, 1].tick_params(axis='x', rotation=45)
    conteo = Counter(emociones)
    axs[2, 0].bar(conteo.keys(), conteo.values(), color='purple'); axs[2, 0].set_title(" Emociones"); axs[2, 0].tick_params(axis='x', rotation=45)
    axs[2, 1].axis('off')
    fig.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ---------------------------------------------
# 6. Callbacks de Gradio
# ---------------------------------------------
def crear_usuario(uid, pw, state_uid):
    if not uid.strip() or not pw.strip():
        return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "", [], "❌ Por favor ingresa usuario y contraseña.")
    profiles = load_profiles()
    if uid in profiles:
        return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "", [], "❌ Usuario ya existe.")
    profiles[uid] = {"password": hash_password(pw), "entries": []}
    save_profiles(profiles)
    return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), uid, [], "")

def iniciar_sesion(uid, pw, state_uid):
    if not uid.strip() or not pw.strip():
        return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "", [], "❌ Por favor ingresa usuario y contraseña.")
    profiles = load_profiles()
    if uid not in profiles:
        return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "", [], "❌ No existe ese usuario.")
    if profiles[uid]["password"] != hash_password(pw):
        return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "", [], "❌ Contraseña incorrecta.")
    if profiles[uid]["entries"]:
        full = " ".join(profiles[uid]["entries"])
        latest = profiles[uid]["entries"][-1]
        _, mensaje = clasificar_y_recomendar(full, latest)
        welcome = [{"role": "assistant", "content": mensaje}]
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), uid, welcome, "")
    else:
        return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), uid, [], "")

def enviar_preguntas(sueño, emocion, cafe, ejercicio, estres, uid):
    if not all(field.strip() for field in [sueño, emocion, cafe, ejercicio, estres]):
        return (gr.update(visible=True), gr.update(visible=False), [], "❌ Por favor completa todos los campos.")
    profiles = load_profiles()
    entry = f"Sueño:{sueño};Emoción:{emocion};Café:{cafe};Ejercicio:{ejercicio};Estres:{estres}"
    profiles[uid]["entries"].append(entry)
    save_profiles(profiles)
    full = " ".join(profiles[uid]["entries"])
    latest = profiles[uid]["entries"][-1]
    _, mensaje = clasificar_y_recomendar(full, latest)
    welcome = [{"role": "assistant", "content": mensaje}]
    return (gr.update(visible=False), gr.update(visible=True), welcome, "")

def chat_submit(msg, history, uid):
    if not msg:
        return "", history
    profiles = load_profiles()
    profiles[uid]["entries"].append(msg)
    save_profiles(profiles)
    full = " ".join(profiles[uid]["entries"])
    latest = profiles[uid]["entries"][-1]
    _, mensaje = clasificar_y_recomendar(full, latest)
    history.append({"role": "user", "content": msg})
    history.append({"role": "assistant", "content": mensaje})
    return "", history

# ---------------------------------------------
# 7. UI en Gradio
# ---------------------------------------------
with gr.Blocks(css=".gradio-container{max-width:600px;margin:auto}") as app:
    with gr.Column(visible=True) as login_sec:
        gr.Markdown("## 🔑 Inicia sesión o crea cuenta")
        uid_in = gr.Textbox(label="ID de usuario")
        pw_in  = gr.Textbox(label="Contraseña", type="password")
        btn_in = gr.Button("Iniciar sesión")
        btn_cr = gr.Button("Crear usuario")
        err    = gr.Markdown("", elem_id="login_error")

    with gr.Column(visible=False) as form_sec:
        gr.Markdown("## 2️⃣ Cuéntame sobre tu día")
        sueño_in     = gr.Textbox(label="Sueño (hrs y sensación)")
        emocion_in   = gr.Textbox(label="Emoción")
        cafe_in      = gr.Textbox(label="Café/estimulantes")
        ejercicio_in = gr.Textbox(label="Ejercicio/pausas")
        estres_in    = gr.Textbox(label="Estrés (1–10)")
        btn_form     = gr.Button("Registrar y empezar")
        form_err     = gr.Markdown("", elem_id="form_error")

    with gr.Column(visible=False) as chat_sec:
        gr.Markdown("## 💬 Tu asistente personal")
        chatbot     = gr.Chatbot(type="messages", height=400)
        user_msg_in = gr.Textbox(placeholder="¿Cómo te sientes ahora?")
        btn_send    = gr.Button("Enviar")
        btn_diag    = gr.Button("📊 Ver diagnóstico")
        output_diag = gr.Image(label="Historial emocional")
        user_state  = gr.State("")

    btn_cr.click(crear_usuario, [uid_in, pw_in, user_state], [login_sec, form_sec, chat_sec, user_state, chatbot, err])
    btn_in.click(iniciar_sesion, [uid_in, pw_in, user_state], [login_sec, form_sec, chat_sec, user_state, chatbot, err])
    btn_form.click(enviar_preguntas, [sueño_in, emocion_in, cafe_in, ejercicio_in, estres_in, user_state], [form_sec, chat_sec, chatbot, form_err])
    btn_send.click(chat_submit, [user_msg_in, chatbot, user_state], [user_msg_in, chatbot])
    btn_diag.click(generar_diagnostico_completo, [user_state], [output_diag])

if __name__ == "__main__":
    app.launch(
        inbrowser=True,   # <-- abre el navegador automáticamente
        show_error=True
    )
