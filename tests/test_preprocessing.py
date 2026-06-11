# Author: Andrés Cabrera Alvarado - A01798681
# Author: Andrea Elizabeth Roman Varela - A01749760
# Author: Pablo Alonso Galván - A01748288
# Fecha de creación: 05/06/2026
# Archivo: tests/test_preprocessing.py
# Descripción general: Pruebas unitarias para las funciones de limpieza de texto (preprocessing.py). 
# Verifica la correcta conversión a minúsculas, eliminación de espacios y caracteres especiales, 
# preservación de hashtags y reparación de problemas de codificación (mojibake).

from src.preprocessing import fix_mojibake, clean_text, normalize_whitespace, remove_urls, remove_mentions

# Verifica que la función clean_text convierta correctamente todo el texto a minúsculas.
def test_clean_text_lowercase():
    result = clean_text("QUIERO SER FLACA")
    assert result == result.lower()

# Comprueba que se eliminen los espacios múltiples, saltos de línea y espacios en los extremos del texto.
def test_clean_text_removes_extra_spaces():
    result = clean_text("hola    mundo   texto")
    assert "  " not in result
    assert result == result.strip()

# Asegura que las etiquetas de hashtag (#) se preserven durante la limpieza.
def test_clean_text_keeps_hashtags():
    result = clean_text("quiero bajar de peso #thinspo")
    assert "#thinspo" in result

# Verifica la eliminación de caracteres de puntuación innecesarios (ruido) que no aportan valor semántico al modelo.
def test_clean_text_removes_noise_characters():
    result = clean_text("hola!!! qué pasa??? $$$ %%%")
    assert "!" not in result
    assert "?" not in result
    assert "$" not in result

# Comprueba que la función de corrección de mojibake restaure los
# caracteres especiales mal codificados (acentos o ñ).
def test_fix_mojibake():
    text = "mÃ¡s frÃ­o"
    fixed = fix_mojibake(text)
    assert "más" in fixed or "frío" in fixed

# Asegura que las palabras relevantes con acentos no se eliminen ni se corrompan.
def test_clean_text_preserves_accented_words():
    result = clean_text("me siento triste y además ansiosa")
    assert "además" in result or "ademas" in result  # puede normalizar o conservar
    assert "triste" in result

# Verifica que una cadena vacía se maneje correctamente sin causar errores.
def test_clean_text_empty_string():
    result = clean_text("")
    assert result == ""

# Comprueba explícitamente el manejo de acentos (conservándolos o normalizándolos, dependiendo de la implementación interna).
def test_clean_text_handles_accents():
    result = clean_text("ánimo corazón")
    assert "á" in result or "animo" in result
    assert "ó" in result or "corazon" in result

# Asegura que los saltos de línea (\n) y tabuladores (\t) se conviertan
# correctamente en espacios regulares.
def test_clean_text_handles_newlines_and_tabs():
    result = clean_text("hola\nmundo\testoy aquí")
    assert "\n" not in result
    assert "\t" not in result

# Verifica que textos extremadamente cortos no rompan la lógica de limpieza.
def test_clean_text_very_short_text():
    result = clean_text("ok")
    assert isinstance(result, str)
    assert len(result) >= 0