# Author: Andrés Cabrera Alvarado - A01798681
# Fecha de creación: 05/06/2026
# Archivo: src/llm_smoke_test.py
# Descripción general: Script de prueba rápida (smoke test) para verificar
# el correcto funcionamiento de la clasificación por LLM con ejemplos básicos.

from src.llm_classifier import classify_text


# Ejecuta la clasificación de una serie de ejemplos predefinidos
# e imprime el resultado en consola para verificar su funcionalidad.
def main():
    examples = [
        "hoy desayuné con mi familia y luego fui a la escuela",
        "llevo dos días ayunando porque me siento enorme",
        "me gustó salir a cenar con mis amigos y me sentí tranquila",
        "quiero bajar de peso dejando de comer por completo",
        "estoy cansada de mi cuerpo y quisiera desaparecer la grasa",
        "hola, mañana tengo examen y estoy nerviosa"
    ]

    for text in examples:
        result = classify_text(text)
        print("\nTEXTO:", text)
        print("RESULTADO:", result)


if __name__ == "__main__":
    main()