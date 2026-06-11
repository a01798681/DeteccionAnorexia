# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 05/06/2026
# Archivo: src/llm_prompts.py
# Descripción general: Construye y formatea el prompt dinámico que se enviará
#   al modelo de lenguaje (LLM). Incluye instrucciones del sistema, formato de
#   salida esperado (JSON) e inyecta ejemplos proporcionados para el few-shot prompting.

def build_dynamic_prompt(text: str, examples: list[dict]) -> str:
    intro = """
Eres un clasificador de textos para detectar señales asociadas a anorexia en publicaciones de redes sociales.

Clasifica cada texto en una de estas dos clases:
- "anorexia": si contiene señales claras asociadas a anorexia, restricción alimentaria, preocupación extrema por el peso o la imagen corporal, ayuno, purga o lenguaje de riesgo.
- "control": si no contiene señales claras asociadas a anorexia.

Responde únicamente en formato JSON con esta estructura:
{
  "label": "anorexia" o "control",
  "risk_score": número entre 0 y 1,
  "reason": "explicación breve"
}

Ejemplos de referencia:
""".strip()

    blocks = []
    for ex in examples:
        risk_score = "0.95" if ex["label"] == "anorexia" else "0.05"
        reason = (
            "El texto muestra señales de restricción alimentaria o preocupación corporal de riesgo."
            if ex["label"] == "anorexia"
            else "El texto no muestra señales claras asociadas a anorexia."
        )

        blocks.append(
            f'Texto: "{ex["text"]}"\n'
            f'Respuesta:\n'
            f'{{"label":"{ex["label"]}","risk_score":{risk_score},"reason":"{reason}"}}'
        )

    ending = f'\n\nAhora clasifica el siguiente texto.\n\nTexto: "{text}"\nRespuesta:'

    return intro + "\n\n" + "\n\n".join(blocks) + ending