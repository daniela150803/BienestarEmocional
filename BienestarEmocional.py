import os
import json
import hashlib
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import gradio as gr
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import concurrent.futures
import time
import matplotlib.pyplot as plt
from PIL import Image
import io
from datetime import datetime
import re

# ---------------------------------------------
# 1. Configuración inicial
# ---------------------------------------------
# Modelo de sentimiento en español
try:
    sentiment_analyzer = pipeline(
        "text-classification",
        model="finiteautomata/beto-sentiment-analysis",
        device="cpu"
    )
    print("✅ Modelo de sentimiento en español cargado")
except Exception as e:
    print(f"❌ Error al cargar modelo de sentimiento: {e}")
    sentiment_analyzer = None

# Configuración de búsqueda web
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "tu_api_key")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID", "tu_engine_id")

# ---------------------------------------------
# 2. Base de conocimiento empática mejorada
# ---------------------------------------------
WELLNESS_KNOWLEDGE = {
    "positivo": {
        "respuestas": [
            "¡Me alegra que te sientas bien! Basado en lo que compartes:",
            "¡Excelente estado de ánimo! Algunas ideas para mantenerlo:",
            "Veo que estás en un buen momento. Recomendaciones:"
        ],
        "recursos": ["bienestar emocional", "hábitos saludables", "crecimiento personal"],
        "follow_ups": [
            "¿Quieres profundizar en alguna estrategia específica?",
            "¿Te interesa más información sobre algún aspecto?",
            "¿Cómo puedo apoyarte para mantener este estado?"
        ],
        "consejos": [
            "🌟 Practica gratitud diaria: anota 3 cosas positivas de tu día",
            "🧘 Mantén rutinas que te beneficien: son clave para el bienestar",
            "🤝 Conecta con otros: las relaciones positivas fortalecen tu estado"
        ]
    },
    "negativo": {
        "respuestas": [
            "Entiendo que estás pasando por un momento difícil. Algunas sugerencias:",
            "Lamento que estés así. Estas estrategias podrían ayudarte:",
            "Por lo que describes, te recomendaría:"
        ],
        "recursos": ["manejo emociones", "apoyo psicológico", "crisis emocional"],
        "follow_ups": [
            "¿Quieres que desarrolle alguna de estas sugerencias?",
            "¿Te gustaría explorar más alternativas?",
            "¿Cómo te sientes con estas recomendaciones?"
        ],
        "consejos": [
            "📝 Escribe sobre tus emociones: puede ayudar a procesarlas",
            "🌿 Sal a caminar aunque sea breve: movimiento y naturaleza ayudan",
            "☎️ Habla con alguien de confianza: el apoyo social es fundamental"
        ],
        "emergencia": [
            "🔴 Detecté que estás en un momento muy difícil. Hay ayuda disponible:",
            "Línea de prevención del suicidio: 911",
            "No estás solo/a, hay profesionales que pueden ayudarte"
        ]
    },
    "neutral": {
        "respuestas": [
            "Por lo que compartes, te sugiero explorar:",
            "Algunas ideas que podrían ser útiles:",
            "Basado en tu situación, considera:"
        ],
        "recursos": ["autoconocimiento", "equilibrio emocional", "bienestar mental"],
        "follow_ups": [
            "¿Alguna de estas sugerencias te parece relevante?",
            "¿Quieres que amplíe alguna recomendación?",
            "¿Cómo podríamos adaptar estas ideas a tu caso?"
        ],
        "consejos": [
            "🧠 Practica mindfulness 5 minutos al día: aumenta conciencia emocional",
            "⏰ Establece metas pequeñas: los logros generan progreso",
            "💬 Reflexiona: ¿qué necesitas ahora para sentirte mejor?"
        ]
    }
}

TOPIC_ADVICE = {
    "estrés": {
        "preguntas": [
            "¿Qué situaciones específicas te generan más estrés?",
            "¿Cómo afecta esto a tu vida diaria?",
            "¿Has notado patrones en cuándo aparece el estrés?"
        ],
        "consejos": [
            "💨 Respiración 4-7-8: inhala 4 seg, mantén 7, exhala 8. Repite",
            "📝 Organiza preocupaciones: separa lo urgente de lo importante",
            "🌳 Pausas activas: cada 90 minutos, descansa 5 minutos",
            "🎵 Escucha música relajante (60 bpm)",
            "🛑 Establece límites saludables: aprende a decir 'no'"
        ]
    },
    "ansiedad": {
        "preguntas": [
            "¿Cómo se manifiesta físicamente tu ansiedad?",
            "¿Qué pensamientos la acompañan?",
            "¿Hay situaciones que hayas empezado a evitar?"
        ],
        "consejos": [
            "🖐️ Técnica 5-4-3-2-1: observa 5 objetos, 4 sonidos, 3 texturas...",
            "📓 Registra patrones: anota cuándo aparece y qué ayuda",
            "🌬️ Respiración diafragmática: mano en abdomen al respirar",
            "🧊 Estimulación térmica: sostén hielo para redirigir atención",
            "🏃 Ejercicio aeróbico: 20 minutos diarios reducen síntomas"
        ]
    },
    "depresión": {
        "preguntas": [
            "¿Cómo ha cambiado tu rutina recientemente?",
            "¿Qué actividades solías disfrutar?",
            "¿Hay momentos del día en que te sientes algo mejor?"
        ],
        "consejos": [
            "🌅 Rutina matutina: levántate a misma hora, sin excepciones",
            "📞 Contacto social: programa al menos una interacción diaria",
            "☀️ Luz solar: 15-30 minutos en la mañana regulan tu ritmo",
            "🎨 Actividad creativa: dibuja, escribe o haz manualidades",
            "🐶 Cuidado de mascotas/plantas: genera responsabilidad"
        ]
    },
    "sueño": {
        "preguntas": [
            "¿Qué pensamientos te mantienen despierto?",
            "¿Cómo es tu rutina antes de dormir?",
            "¿Has notado patrones en tus dificultades?"
        ],
        "consejos": [
            "🛌 Higiene del sueño: usa la cama solo para dormir",
            "🌡️ Temperatura ideal: 18-21°C en la habitación",
            "📵 Desconexión digital: evita pantallas 1 hora antes",
            "🍵 Infusiones relajantes: manzanilla, valeriana...",
            "📖 Lectura ligera: 20 minutos de libro físico"
        ]
    }
}

DAILY_WELLNESS_TEST = {
    "questions": [
        {
            "text": "¿Cómo calificarías tu estado de ánimo hoy?",
            "options": ["😢 Muy bajo", "😞 Bajo", "😐 Neutral", "🙂 Bueno", "😊 Excelente"],
            "key": "mood"
        },
        {
            "text": "¿Cómo ha sido tu nivel de energía?",
            "options": ["😴 Sin energía", "😪 Baja", "🫤 Normal", "😌 Buena", "💪 Alta"],
            "key": "energy"
        },
        {
            "text": "¿Cómo describirías tus pensamientos?",
            "options": ["😣 Muy negativos", "😕 Algo negativos", "🤔 Neutrales", "😊 Positivos", "😁 Muy positivos"],
            "key": "thoughts"
        },
        {
            "text": "¿Qué nivel de estrés/ansiedad sentiste?",
            "options": ["😫 Extremo", "😰 Mucho", "😐 Moderado", "😌 Poco", "😊 Nada"],
            "key": "stress"
        },
        {
            "text": "¿Cómo fue tu interacción social?",
            "options": ["🚫 Nula", "👥 Poca", "🤝 Normal", "💬 Buena", "❤️ Excelente"],
            "key": "social"
        }
    ],
    "scoring": {
        "mood": {"weights": [-2, -1, 0, 1, 2]},
        "energy": {"weights": [-2, -1, 0, 1, 2]},
        "thoughts": {"weights": [-2, -1, 0, 1, 2]},
        "stress": {"weights": [2, 1, 0, -1, -2]},
        "social": {"weights": [-1, -0.5, 0, 0.5, 1]}
    }
}

# ---------------------------------------------
# 3. Sistema de respuesta empática mejorado
# ---------------------------------------------
class EmpathicWellnessAssistant:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.topics = list(TOPIC_ADVICE.keys())
        self.vectorizer.fit(self.topics)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.user_states = defaultdict(dict)
        self.advice_cache = defaultdict(list)
        self.last_query_time = 0
        self.min_query_interval = 3600  # 1 hora entre búsquedas similares
        self.trusted_sources = [
            "site:who.int",
            "site:mayoclinic.org",
            "site:medlineplus.gov",
            "site:clinicabarcelona.org",
            "site:webconsultas.com",
            "site:psicologiaymente.com"
        ]

    def analyze_sentiment(self, text: str) -> str:
        """Analiza el sentimiento del texto en español"""
        if not sentiment_analyzer or not text.strip():
            return "neutral"

        try:
            result = sentiment_analyzer(text[:512])[0]
            return "positivo" if result['label'] == 'POS' else "negativo"
        except Exception as e:
            print(f"Error en análisis de sentimiento: {e}")
            return "neutral"

    def get_professional_advice(self, topic: str, user_context: str) -> List[Dict]:
        """Obtiene consejos de fuentes confiables"""
        try:
            cache_key = f"{topic}_{hashlib.md5(user_context.encode()).hexdigest()[:6]}"
            if cache_key in self.advice_cache and time.time() - self.last_query_time < self.min_query_interval:
                return self.advice_cache[cache_key]

            query = f"{topic} {user_context} {' OR '.join(self.trusted_sources)}"

            params = {
                "key": SEARCH_API_KEY,
                "cx": SEARCH_ENGINE_ID,
                "q": query,
                "num": 5,
                "lr": "lang_es",
                "sort": "date"
            }

            response = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)
            results = []

            if response.status_code == 200:
                for item in response.json().get('items', []):
                    if any(ext in item.get('link', '') for ext in ['.pdf', '.doc']):
                        continue

                    snippet = self._clean_text(item.get('snippet', ''))
                    if len(snippet) > 100:
                        results.append({
                            "title": self._clean_title(item.get('title', 'Recurso profesional')),
                            "url": item.get('link', '#'),
                            "snippet": snippet[:250],
                            "source": self._extract_source(item.get('link', ''))
                        })

                self.advice_cache[cache_key] = results
                self.last_query_time = time.time()
                return results

        except Exception as e:
            print(f"Error en búsqueda web: {str(e)[:200]}")

        return []

    def _clean_text(self, text: str) -> str:
        """Limpia texto de formato extraño"""
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _clean_title(self, title: str) -> str:
        """Limpia títulos de resultados"""
        return title.split('|')[0].split('-')[0].strip()

    def _extract_source(self, url: str) -> str:
        """Extrae el nombre de la fuente de la URL"""
        domain = url.split('/')[2] if len(url.split('/')) > 2 else url
        return domain.replace('www.', '').split('.')[0].capitalize()

    def format_professional_advice(self, advice_list: List[Dict]) -> str:
        """Formatea los consejos profesionales para mostrarlos"""
        if not advice_list:
            return ""

        response = ["\n\n💡 Basado en información profesional:"]
        seen_sources = set()

        for i, advice in enumerate(advice_list[:3], 1):
            if advice['source'] not in seen_sources:
                response.append(
                    f"{i}. [{advice['title']}]({advice['url']}) "
                    f"(Fuente: {advice['source']})\n"
                    f"   {advice['snippet']}"
                )
                seen_sources.add(advice['source'])

        return "\n".join(response)

    def generate_response(self, user_input: str, user_id: str, history: list) -> str:
        """Genera una respuesta con apoyo emocional personalizado"""
        profiles = load_profiles()
        user_data = profiles.get(user_id, {})

        # 1. Análisis de sentimiento
        sentiment = self.analyze_sentiment(user_input)

        # 2. Manejo de primera interacción
        if len(history) < 2:
            return (
                f"¡Hola {user_id}! Para ayudarte mejor:\n"
                "1. ¿Qué situación te preocupa principalmente?\n"
                "2. ¿Cómo ha sido tu estado emocional esta semana?\n"
                "3. ¿Qué has intentado hasta ahora?"
            )

        # 3. Manejo de respuestas cortas
        if len(user_input.split()) < 6:
            last_topic = self.user_states[user_id].get("last_topic")
            if last_topic:
                question = np.random.choice(TOPIC_ADVICE[last_topic]["preguntas"])
                return f"Para ayudarte mejor con {last_topic}, {question.lower()}"
            return np.random.choice([
                "¿Podrías compartir más detalles para entender mejor?",
                "¿Puedes contarme un poco más sobre la situación?",
                "¿Qué aspecto te afecta más actualmente?"
            ])

        # 4. Construir respuesta
        response_parts = []

        # Añadir reconocimiento emocional
        response_parts.append(np.random.choice(WELLNESS_KNOWLEDGE[sentiment]["respuestas"]))

        # 5. Añadir consejos generales
        general_advice = np.random.choice(
            WELLNESS_KNOWLEDGE[sentiment]["consejos"],
            size=min(2, len(WELLNESS_KNOWLEDGE[sentiment]["consejos"])),
            replace=False
        )
        response_parts.append("\n\n🔹 " + "\n🔹 ".join(general_advice))

        # 6. Detección de temas específicos
        user_vec = self.vectorizer.transform([user_input])
        topic_vecs = self.vectorizer.transform(self.topics)
        sims = cosine_similarity(user_vec, topic_vecs)[0]
        best_idx = np.argmax(sims)

        if sims[best_idx] > 0.4:
            topic = self.topics[best_idx]
            self.user_states[user_id]["last_topic"] = topic

            # Añadir consejos específicos
            specific_advice = np.random.choice(
                TOPIC_ADVICE[topic]["consejos"],
                size=min(2, len(TOPIC_ADVICE[topic]["consejos"])),
                replace=False
            )
            response_parts.append(f"\n\n📌 Para {topic.capitalize()}:\n⭐ " + "\n⭐ ".join(specific_advice))

            # Añadir consejos profesionales
            professional_advice = self.get_professional_advice(topic, user_input)
            if professional_advice:
                response_parts.append(self.format_professional_advice(professional_advice))

        # 7. Manejo de crisis
        crisis_words = ["suicidio", "morir", "no puedo más", "sin esperanza", "acabar con todo"]
        if (sentiment == "negativo" and
                any(re.search(rf"\b{word}\b", user_input.lower()) for word in crisis_words)):
            return "\n".join(WELLNESS_KNOWLEDGE["negativo"]["emergencia"])

        # 8. Añadir pregunta de seguimiento
        follow_up = self._get_follow_up(sentiment, self.user_states[user_id].get("last_topic"))
        response_parts.append(f"\n\n{follow_up}")

        # 9. Búsqueda de recursos en segundo plano
        if sentiment != "neutral":
            search_query = np.random.choice(WELLNESS_KNOWLEDGE[sentiment]["recursos"])
            self.executor.submit(self._add_web_resources, search_query, user_id, history)

        return "".join(response_parts)

    def _get_follow_up(self, sentiment: str, topic: str = None) -> str:
        """Genera pregunta de seguimiento relevante"""
        if topic and np.random.random() < 0.7:
            return np.random.choice(TOPIC_ADVICE[topic]["preguntas"])
        return np.random.choice(WELLNESS_KNOWLEDGE[sentiment]["follow_ups"])

    def _add_web_resources(self, query: str, user_id: str, history: list):
        """Añade recursos adicionales a la conversación"""
        resources = self.get_professional_advice(query, "")
        if resources and history:
            resource_msg = "\n\n📚 Más recursos útiles:\n"
            seen_sources = set()

            for i, res in enumerate(resources[:2], 1):
                if res['source'] not in seen_sources:
                    resource_msg += (
                        f"{i}. [{res['title']}]({res['url']}) "
                        f"(Fuente: {res['source']})\n"
                        f"   {res['snippet'][:100]}...\n"
                    )
                    seen_sources.add(res['source'])

            history[-1] = (history[-1][0], history[-1][1] + resource_msg)

# ---------------------------------------------
# 4. Gestión de usuarios y progreso
# ---------------------------------------------
PROFILE_PATH = "user_profiles.json"
    
def load_profiles() -> Dict:
    """Carga los perfiles de usuarios desde archivo"""
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_profiles(profiles: Dict):
    """Guarda los perfiles de usuarios en archivo"""
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

def hash_password(pw: str) -> str:
    """Genera hash seguro de contraseña"""
    return hashlib.sha256(pw.encode()).hexdigest()

def show_daily_test(uid: str) -> bool:
    """Determina si mostrar test diario"""
    profiles = load_profiles()
    if uid not in profiles:
        return False

    last_test = profiles[uid].get("last_test_date")
    today = datetime.now().strftime("%Y-%m-%d")
    return last_test != today

def save_test_results(uid: str, answers: dict) -> int:
    """Guarda resultados del test diario"""
    profiles = load_profiles()
    if uid not in profiles:
        return 50

    # Calcular puntuación
    score = 0
    scoring = DAILY_WELLNESS_TEST["scoring"]
    for key, ans_idx in answers.items():
        if key in scoring:
            score += scoring[key]["weights"][ans_idx]

    # Normalizar a porcentaje (0-100)
    max_score = sum(max(w["weights"]) for w in scoring.values())
    min_score = sum(min(w["weights"]) for w in scoring.values())
    normalized = int(((score - min_score) / (max_score - min_score)) * 100)

    # Guardar resultados
    profiles[uid]["last_test_date"] = today = datetime.now().strftime("%Y-%m-%d")
    profiles[uid]["test_history"] = profiles[uid].get("test_history", []) + [{
        "date": datetime.now().isoformat(),
        "answers": answers,
        "score": normalized
    }]
    save_profiles(profiles)
    return normalized


#Submit_test

def submit_test(*answers_and_state):
    """Procesa el test diario"""
    try:
        # Separar respuestas y estado
        *answers, uid = answers_and_state

        if not uid:
            print("Error: user_state vacío en submit_test")
            return {
                test_result: "❌ Sesión no válida. Por favor, inicia sesión nuevamente.",
                test_submit_btn: gr.update(visible=True),
                test_continue_btn: gr.update(visible=False)
            }

        # Convertir respuestas a diccionario
        answers_dict = {}
        for i, q in enumerate(DAILY_WELLNESS_TEST["questions"]):
            if i < len(answers) and answers[i] is not None:
                answers_dict[q["key"]] = q["options"].index(answers[i])
            else:
                answers_dict[q["key"]] = 0  # Valor por defecto

        # Guardar resultados
        score = save_test_results(uid, answers_dict)
        print(f"Test guardado para {uid}. Puntuación: {score}")

        return {
            test_result: f"✅ Test completado. Puntuación: {score}/100",
            test_submit_btn: gr.update(visible=False),
            test_continue_btn: gr.update(visible=True)
        }
    except Exception as e:
        print(f"Error en submit_test: {e}")
        return {
            test_result: f"❌ Error al procesar el test: {str(e)}",
            test_submit_btn: gr.update(visible=True),
            test_continue_btn: gr.update(visible=False)
        }

def generar_grafico_progreso(uid):
    """Genera gráficos de progreso para el usuario"""
    try:
        if not uid:
            print("Error: No se proporcionó UID")
            return generar_imagen_mensaje("No se identificó al usuario")

        profiles = load_profiles()
        if uid not in profiles:
            print(f"Error: Usuario {uid} no encontrado en perfiles")
            return generar_imagen_mensaje("Usuario no registrado")

        user_data = profiles[uid]
        test_history = user_data.get("test_history", [])
        entries = user_data.get("entries", [])

        print(f"Datos encontrados para {uid} - Tests: {len(test_history)}, Entradas: {len(entries)}")

        # Caso sin ningún dato
        if not test_history and not entries:
            return generar_imagen_mensaje("Completa tu primer test diario para ver tu progreso")

        # Configurar figura según qué datos existen
        if test_history and entries:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        else:
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = None

        # Gráfico 1: Progreso del test diario
        if test_history:
            try:
                fechas = []
                datos = {v: [] for v in ["mood", "energy", "thoughts", "stress", "social"]}

                for test in test_history[-14:]:  # Últimos 14 tests
                    date_obj = datetime.fromisoformat(test["date"])
                    fechas.append(date_obj.strftime("%d/%m"))
                    for var in datos.keys():
                        datos[var].append(test["answers"].get(var, 0))

                # Graficar cada variable
                for var, values in datos.items():
                    ax1.plot(fechas, values, marker='o', label=var.capitalize())

                ax1.set_title("Evolución Test Diario de Bienestar")
                ax1.set_ylabel("Puntuación")
                ax1.set_xlabel("Fecha")
                ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
                ax1.grid(True, linestyle='--', alpha=0.7)
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

                # Ajustar límites del eje Y según las opciones del test
                ax1.set_ylim(-0.5, 4.5)
                ax1.set_yticks([0, 1, 2, 3, 4])

            except Exception as e:
                print(f"Error al generar gráfico de tests: {e}")
                ax1.clear()
                ax1.text(0.5, 0.5, 'Error al procesar datos del test',
                         ha='center', va='center', color='red')
                ax1.axis('off')

            # Gráfico 2: Análisis de sentimientos (adaptado)
        if entries and ax2:
            try:
                fechas = []
                sentimientos = []
    
                for entry in entries[-14:]:  # Últimas 14 entradas
                    # Manejar tanto strings viejos como nuevos objetos
                    if isinstance(entry, dict):
                        text = entry.get("content", "")
                        entry_date = entry.get("date", datetime.now().isoformat())
                        sentiment = entry.get("sentiment", "")
                    else:
                        # Formato antiguo (string)
                        text = str(entry)
                        entry_date = datetime.now().isoformat()
                        sentiment = ""
    
                    # Procesar fecha
                    date_obj = datetime.fromisoformat(entry_date) if ":" in entry_date else datetime.now()
                    fechas.append(date_obj.strftime("%d/%m"))
    
                    # Determinar valor numérico
                    if sentiment:
                        # Usar clasificación manual si existe
                        sentimiento = 1 if sentiment == "positivo" else (-1 if sentiment == "negativo" else 0)
                    else:
                        # Analizar con el modelo si no hay clasificación
                        if text.strip():
                            try:
                                result = sentiment_analyzer(text[:512])[0]
                                sentimiento = 1 if result['label'] == 'POS' else (-1 if result['label'] == 'NEG' else 0)
                            except:
                                sentimiento = 0
                        else:
                            sentimiento = 0
    
                    sentimientos.append(sentimiento)

                ax2.plot(fechas, sentimientos, marker='o', color='purple', linestyle='-')
                ax2.fill_between(fechas, sentimientos, 0, color='purple', alpha=0.1)
                ax2.axhline(0, color='gray', linestyle='--')
                ax2.set_title("Análisis de Sentimiento en Conversaciones")
                ax2.set_ylabel("Valoración")
                ax2.set_yticks([-1, 0, 1])
                ax2.set_yticklabels(["Negativo", "Neutral", "Positivo"])
                ax2.grid(True, linestyle='--', alpha=0.5)
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

            except Exception as e:
                print(f"Error al generar gráfico de sentimientos: {e}")
                ax2.clear()
                ax2.text(0.5, 0.5, 'Error al procesar análisis de sentimiento',
                         ha='center', va='center', color='red')
                ax2.axis('off')

        plt.tight_layout()

        # Convertir a imagen para Gradio
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    except Exception as e:
        print(f"Error crítico en generar_grafico_progreso: {e}")
        return generar_imagen_mensaje("Error al generar el gráfico de progreso")

def generar_imagen_mensaje(mensaje):
    """Función auxiliar para generar imágenes con mensajes"""
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(0.5, 0.5, mensaje,
            ha='center', va='center',
            fontsize=12, color='#333333')
    ax.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def show_welcome_messages(uid: str):
    """Muestra mensajes de bienvenida personalizados"""
    profiles = load_profiles()
    user_data = profiles.get(uid, {})

    # Cambiamos a formato de mensajes compatible
    welcome_msg = [
        {"role": "assistant", "content": f"¡Bienvenid{'a' if uid.endswith('a') else 'o'} {uid}! 🌱"},
        {"role": "assistant", "content": "Soy tu asistente de bienestar emocional. Para comenzar:"},
        {"role": "assistant", "content": "1. ¿Qué te trae por aquí hoy?"},
        {"role": "assistant", "content": "2. ¿Cómo describirías tu estado general esta semana?"},
        {"role": "assistant", "content": "3. ¿Hay algo específico que quieras trabajar?"}
    ]

    test_history = user_data.get("test_history", [])
    if test_history:
        last_score = test_history[-1]["score"]
        if last_score < 30:
            welcome_msg.append({"role": "assistant", "content": "Veo que en tu último test estabas pasando un momento difícil. ¿Cómo estás hoy?"})
        elif last_score > 70:
            welcome_msg.append({"role": "assistant", "content": "¡Me alegra que en tu último test estabas bien! ¿Sigues igual?"})
        welcome_msg.append({"role": "assistant", "content": f"Tu última puntuación de bienestar: {last_score}/100"})

    return welcome_msg


assistant = EmpathicWellnessAssistant()

def chat_interaction(msg: str, history: List[Dict], uid: str):
    """Maneja la interacción del chat con formato messages"""
    if not msg.strip() or not uid:
        return "", history

    # Guardar mensaje en el perfil
    profiles = load_profiles()
    if uid in profiles:
        profiles[uid]["entries"].append(msg)
        save_profiles(profiles)

    # Convertir historia al formato correcto si es necesario
    if history and isinstance(history[0], tuple):
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": content}
                   for i, (content, _) in enumerate(history)]

    # Añadir nuevo mensaje de usuario
    history.append({"role": "user", "content": msg})

    # Generar respuesta
    bot_response = assistant.generate_response(msg, uid, history)
    history.append({"role": "assistant", "content": bot_response})

    return "", history


#Usuarios (Crear y Login)

def register_user(uid: str, pw: str):
    """Registra un nuevo usuario"""
    if not uid.strip() or not pw.strip():
        return {err: "❌ Ingresa usuario y contraseña"}

    profiles = load_profiles()
    if uid in profiles:
        return {err: "❌ Usuario ya existe"}

    # Inicializar todas las estructuras necesarias
    profiles[uid] = {
        "password": hash_password(pw),
        "entries": [],
        "test_history": [],  # Asegurar que existe esta clave
        "created_at": datetime.now().isoformat(),  # Usar datetime en lugar de time
        "last_test_date": None
    }

    try:
        save_profiles(profiles)
        print(f"Usuario {uid} registrado correctamente")  # Log de depuración
    except Exception as e:
        print(f"Error al guardar perfil: {e}")
        return {err: "❌ Error al crear el perfil"}

    show_test = True  # Forzar mostrar el test para nuevos usuarios
    return {
        login_sec: gr.update(visible=False),
        test_sec: gr.update(visible=show_test),
        chat_sec: gr.update(visible=not show_test),
        user_state: uid,
        err: ""
    }

def login_user(uid: str, pw: str):
    """Inicia sesión de usuario"""
    profiles = load_profiles()
    if uid not in profiles or profiles[uid]["password"] != hash_password(pw):
        return {err: "❌ Usuario/clave inválidos"}

    show_test = show_daily_test(uid)
    return {
        login_sec: gr.update(visible=False),
        test_sec: gr.update(visible=show_test),
        chat_sec: gr.update(visible=not show_test),
        user_state: uid,
        err: ""
    }

# ---------------------------------------------
# 5. Interfaz de usuario
# ---------------------------------------------

css = """
.gradio-container { max-width: 750px; margin: auto; }
.message.user { 
    background: #e3f2fd; 
    border-radius: 18px 18px 0 18px;
    padding: 12px 16px;
    margin-left: 15%;
    margin-bottom: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.message.bot { 
    background: #f8f9fa; 
    border-radius: 18px 18px 18px 0;
    padding: 12px 16px;
    margin-right: 15%;
    margin-bottom: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.dark .message.user { background: #0d47a1; color: white; }
.dark .message.bot { background: #424242; color: white; }
#chatbot { min-height: 500px; }
.emergency { color: #d32f2f; font-weight: bold; }
.progress-container { margin-top: 20px; }
.test-question { margin-bottom: 15px; }
.test-options { display: flex; flex-direction: column; gap: 8px; }
.resource-link { color: #1a73e8; text-decoration: underline; }
"""

with gr.Blocks(css=css) as app:
    user_state = gr.State("")

    # Sección de login/registro
    with gr.Column(visible=True) as login_sec:
        gr.Markdown("## 🌟 Asistente de Bienestar Emocional")
        gr.Markdown("Tu espacio confidencial para apoyo emocional")
        with gr.Row():
            uid_in = gr.Textbox(label="Usuario", placeholder="Tu nombre o alias")
            pw_in = gr.Textbox(label="Contraseña", type="password")
        with gr.Row():
            btn_in = gr.Button("Iniciar sesión", variant="primary")
            btn_cr = gr.Button("Registrarse", variant="secondary")
        err = gr.Markdown("")

    # Sección de test diario
    with gr.Column(visible=False) as test_sec:
        gr.Markdown("## 📝 Test Diario de Bienestar")
        gr.Markdown("Por favor responde estas breves preguntas:")

        test_answers = []
        for i, question in enumerate(DAILY_WELLNESS_TEST["questions"]):
            with gr.Group():
                gr.Markdown(f"**{i+1}. {question['text']}**")
                answer = gr.Radio(choices=question["options"], label="Selecciona")
                test_answers.append(answer)

        test_submit_btn = gr.Button("Enviar respuestas", variant="primary")
        test_result = gr.Markdown("")
        test_continue_btn = gr.Button("Continuar al chat", visible=False)

    # Sección de chat principal
    with gr.Column(visible=False) as chat_sec:
        gr.Markdown("## 💬 Conversación de Apoyo")
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            show_label=False,
            height=400,
            type="messages"  # Formato actualizado
        )
        user_in = gr.Textbox(placeholder="¿Cómo te sientes hoy?...", lines=3)
        with gr.Row():
            btn_send = gr.Button("Enviar", variant="primary")
            btn_clear = gr.Button("Limpiar", variant="secondary")
            btn_progress = gr.Button("Ver progreso", variant="secondary")

        with gr.Column(visible=False) as progress_sec:
            progress_graph = gr.Image(label="Tu progreso", interactive=False)

        gr.Markdown("*Este asistente no sustituye ayuda profesional*")

    # Funciones de flujo
    def mostrar_y_generar(uid):
        return gr.update(visible=True), generar_grafico_progreso(uid)

    # Conexiones de eventos
    btn_cr.click(
        register_user,
        [uid_in, pw_in],
        [login_sec, test_sec, chat_sec, user_state, err]
    ).then(
        lambda uid: show_welcome_messages(uid) if not show_daily_test(uid) else [],
        [user_state],
        [chatbot]
    )

    btn_in.click(
        login_user,
        [uid_in, pw_in],
        [login_sec, test_sec, chat_sec, user_state, err]
    ).then(
        lambda uid: show_welcome_messages(uid) if not show_daily_test(uid) else [],
        [user_state],
        [chatbot]
    )

    test_submit_btn.click(
        submit_test,
        test_answers + [user_state],  # Envía tanto las respuestas como el estado
        [test_result, test_submit_btn, test_continue_btn]
    )

    test_continue_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[test_sec, chat_sec]
    ).then(
        lambda uid: show_welcome_messages(uid),
        [user_state],
        [chatbot]
    )

    btn_send.click(
        chat_interaction,
        [user_in, chatbot, user_state],
        [user_in, chatbot]
    )
    user_in.submit(
        chat_interaction,
        [user_in, chatbot, user_state],
        [user_in, chatbot]
    )

    btn_clear.click(lambda: [], None, chatbot)
    btn_progress.click(
        mostrar_y_generar,
        inputs=[user_state],
        outputs=[progress_sec, progress_graph]
    )

if __name__ == "__main__":
    app.launch(
        inbrowser=True,
        show_error=True
    )
