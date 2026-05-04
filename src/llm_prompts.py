FEW_SHOT_PROMPT = """
Eres un clasificador de textos para detectar señales asociadas a anorexia en publicaciones de redes sociales.

Clasifica cada texto en una de estas dos clases:
- "anorexia": si contiene señales claras asociadas a anorexia, restricción alimentaria, preocupación extrema por el peso o la imagen corporal, ayuno, purga o lenguaje de riesgo.
- "control": si no contiene señales claras asociadas a anorexia.

Responde únicamente en formato JSON con esta estructura:
{{
  "label": "anorexia" o "control",
  "risk_score": número entre 0 y 1,
  "reason": "explicación breve"
}}

Ejemplos:

Texto: "quiero dejar de comer hasta ser flaca"
Respuesta:
{{"label":"anorexia","risk_score":0.97,"reason":"El texto expresa restricción alimentaria y deseo extremo de delgadez."}}

Texto: "me siento bien y salí a comer con mis amigos"
Respuesta:
{{"label":"control","risk_score":0.06,"reason":"No hay señales claras de anorexia y el contexto es cotidiano."}}

Texto: "hoy no quiero comer nada porque me siento gorda"
Respuesta:
{{"label":"anorexia","risk_score":0.95,"reason":"El texto relaciona rechazo a la comida con preocupación corporal."}}

Texto: "hola, que tal todo?"
Respuesta:
{{"label":"control","risk_score":0.03,"reason":"Es un texto casual sin contenido de riesgo."}}

Ahora clasifica el siguiente texto.

Texto: "{text}"
Respuesta:
"""