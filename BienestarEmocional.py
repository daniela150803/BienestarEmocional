import os, json, hashlib, requests
import gradio as gr
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
# 2. API de reglas en línea
# ---------------------------------------------
RULES_API = "https://tu-dominio.com/api/rules"

def fetch_rules():
    """Descarga el listado de reglas desde el servicio REST."""
    try:
        resp = requests.get(RULES_API, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"⚠️ No pude obtener reglas: {e}")
        return []

def eval_rules(estado):
    """
    Dado un estado dict con keys como 'sleep_hours','stress_level','emotion',
    devuelve lista de consejos {'advice','source'}.
    """
    recomendaciones = []
    reglas = fetch_rules()
    for regla in reglas:
        cond = regla.get("condition", {})
        cumple = True
        for var, expr in cond.items():
            # expr es string como '<6' o '>7'
            op, umbral = expr[0], float(expr[1:])
            val = estado.get(var, 0)
            if op == "<" and not (val < umbral): cumple=False
            if op == ">" and not (val > umbral): cumple=False
            if not cumple: break
        if cumple:
            recomendaciones.append({
                "advice": regla.get("advice",""),
                "source": regla.get("source","")
            })
    return recomendaciones

# ---------------------------------------------
# 3. Perfil de usuario
# ---------------------------------------------
PROFILE_PATH = "user_profiles.json"
def load_profiles():
    if os.path.exists(PROFILE_PATH):
        return json.load(open(PROFILE_PATH,"r"))
    return {}
def save_profiles(profiles):
    with open(PROFILE_PATH,"w") as f:
        json.dump(profiles,f,indent=2)
def hash_password(pw:str)->str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ---------------------------------------------
# 4. Clasificación + generación de consejos
# ---------------------------------------------
def clasificar_y_recomendar(full_text:str, latest_entry:str):
    # 1) Sentimiento
    if not sentiment_pipeline:
        return [], "Servicio no disponible"
    res = sentiment_pipeline(full_text[:512])[0]
    score = int(res["label"].split()[0])
    category = ("Necesita apoyo","Neutral","Bienestar alto")[min(score-1,2)]
    # 2) Extraer métricas
    parts = {}
    for p in latest_entry.split(";"):
        if ":" in p:
        k, v = p.split(":", 1)
        parts[k] = v
    try:
        sleep_hours = float(parts.get("Sueño","0").split()[0])
    except:
        sleep_hours = 0.0
    try:
        stress_level = int(parts.get("Estres","0"))
    except:
        stress_level = 0
    emotion = parts.get("Emoción","").strip()
    estado = {
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "emotion": emotion
    }
    # 3) Evaluar reglas remotas
    recs = eval_rules(estado)
    # 4) Construir salida
    header = f"💬 *Puntuación* {score}/5 → **{category}**\n"
    advice_lines = "\n".join(f"- {r['advice']} _(fuente: {r['source']})_" for r in recs)
    if not advice_lines:
        advice_lines = "Por ahora no hay consejos nuevos, ¡hablemos mañana!"
    return recs, header + advice_lines

# ---------------------------------------------
# 5. Callbacks de Gradio
# ---------------------------------------------
def crear_usuario(uid,pw,state_uid):
    profiles = load_profiles()
    if uid in profiles:
        return (
            gr.update(visible=True),  # sigue login
            gr.update(visible=False),
            gr.update(visible=False),
            "", [], "❌ Usuario existe."
        )
    profiles[uid] = {"password":hash_password(pw),"entries":[]}
    save_profiles(profiles)
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        uid,[], ""
    )

def iniciar_sesion(uid,pw,state_uid):
    profiles = load_profiles()
    if uid not in profiles:
        return (gr.update(visible=True),gr.update(visible=False),gr.update(visible=False),"",[],"❌ No existe usuario")
    if profiles[uid]["password"]!=hash_password(pw):
        return (gr.update(visible=True),gr.update(visible=False),gr.update(visible=False),"",[],"❌ Contraseña incorrecta")
    if profiles[uid]["entries"]:
        # ya respondió inicial → recomendaciones inmediatas
        full = " ".join(profiles[uid]["entries"])
        latest = profiles[uid]["entries"][-1]
        _, mensaje = clasificar_y_recomendar(full,latest)
        welcome = [{"role":"assistant","content":mensaje}]
        return (gr.update(visible=False),gr.update(visible=False),gr.update(visible=True),uid,welcome,"")
    else:
        return (gr.update(visible=False),gr.update(visible=True),gr.update(visible=False),uid,[],"")

def enviar_preguntas(sueño,emocion,cafe,ejercicio,estres,uid):
    profiles = load_profiles()
    entry = f"Sueño:{sueño};Emoción:{emocion};Café:{cafe};Ejercicio:{ejercicio};Estres:{estres}"
    profiles[uid]["entries"].append(entry)
    save_profiles(profiles)
    full = " ".join(profiles[uid]["entries"])
    latest = profiles[uid]["entries"][-1]
    _, mensaje = clasificar_y_recomendar(full,latest)
    welcome = [{"role":"assistant","content":mensaje}]
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        welcome
    )

def chat_submit(msg,history,uid):
    if not msg: return "",history
    profiles = load_profiles()
    profiles[uid]["entries"].append(msg)
    save_profiles(profiles)
    full = " ".join(profiles[uid]["entries"])
    latest = profiles[uid]["entries"][-1]
    _, mensaje = clasificar_y_recomendar(full,latest)
    history.append({"role":"user","content":msg})
    history.append({"role":"assistant","content":mensaje})
    return "",history

# ---------------------------------------------
# 6. UI en Gradio
# ---------------------------------------------
with gr.Blocks(css=".gradio-container{max-width:600px;margin:auto}") as app:
    # — Login / Registro —
    with gr.Column(visible=True) as login_sec:
        gr.Markdown("## 🔑 Inicia sesión o crea cuenta")
        uid_in = gr.Textbox(label="ID de usuario")
        pw_in  = gr.Textbox(label="Contraseña",type="password")
        btn_in = gr.Button("Iniciar sesión")
        btn_cr = gr.Button("Crear usuario")
        err    = gr.Markdown("",elem_id="login_error")
    # — Formulario inicial —
    with gr.Column(visible=False) as form_sec:
        gr.Markdown("## 2️⃣ Cuéntame sobre tu día")
        sueño_in   = gr.Textbox(label="Sueño (hrs y sensación)")
        emocion_in = gr.Textbox(label="Emoción")
        cafe_in    = gr.Textbox(label="Café/estimulantes")
        ejercicio_in=gr.Textbox(label="Ejercicio/pausas")
        estres_in  = gr.Textbox(label="Estrés (1–10)")
        btn_form   = gr.Button("Registrar y empezar")
    # — Chat —
    with gr.Column(visible=False) as chat_sec:
        gr.Markdown("## 💬 Tu asistente personal")
        chatbot     = gr.Chatbot(type="messages",height=400)
        user_msg_in = gr.Textbox(placeholder="¿Cómo te sientes ahora?")
        btn_send    = gr.Button("Enviar")
    user_state = gr.State("")

    btn_cr.click(
        crear_usuario,
        [uid_in,pw_in,user_state],
        [login_sec,form_sec,chat_sec,user_state,chatbot,err]
    )
    btn_in.click(
        iniciar_sesion,
        [uid_in,pw_in,user_state],
        [login_sec,form_sec,chat_sec,user_state,chatbot,err]
    )
    btn_form.click(
        enviar_preguntas,
        [sueño_in,emocion_in,cafe_in,ejercicio_in,estres_in,user_state],
        [form_sec,chat_sec,chatbot]
    )
    btn_send.click(
        chat_submit,
        [user_msg_in,chatbot,user_state],
        [user_msg_in,chatbot]
    )

if __name__ == "__main__":
    app.launch(
        inbrowser=True,   # <-- abre el navegador automáticamente
        show_error=True
    )
