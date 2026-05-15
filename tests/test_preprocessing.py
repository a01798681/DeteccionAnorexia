from src.preprocessing import fix_mojibake, clean_text, normalize_whitespace, remove_urls, remove_mentions

#Convierte a minúsculas
def test_clean_text_lowercase():
    result = clean_text("QUIERO SER FLACA")
    assert result == result.lower()

#Elimina espacios extra
def test_clean_text_removes_extra_spaces():
    result = clean_text("hola    mundo   texto")
    assert "  " not in result
    assert result == result.strip()

#Conserva hashtags
def test_clean_text_keeps_hashtags():
    result = clean_text("quiero bajar de peso #thinspo")
    assert "#thinspo" in result

#Elimina caracteres de ruido
def test_clean_text_removes_noise_characters():
    result = clean_text("hola!!! qué pasa??? $$$ %%%")
    assert "!" not in result
    assert "?" not in result
    assert "$" not in result

#Corrige codificación rara (mojibake)
def test_fix_mojibake():
    text = "mÃ¡s frÃ­o"
    fixed = fix_mojibake(text)
    assert "más" in fixed or "frío" in fixed

#Preserva palabras importantes (letras con acentos)
def test_clean_text_preserves_accented_words():
    result = clean_text("me siento triste y además ansiosa")
    assert "además" in result or "ademas" in result  # puede normalizar o conservar
    assert "triste" in result

#Maneja texto vacío
def test_clean_text_empty_string():
    result = clean_text("")
    assert result == ""

#Maneja texto con acentos
def test_clean_text_handles_accents():
    result = clean_text("ánimo corazón")
    assert "á" in result or "animo" in result
    assert "ó" in result or "corazon" in result

#Maneja saltos de línea y tabs
def test_clean_text_handles_newlines_and_tabs():
    result = clean_text("hola\nmundo\testoy aquí")
    assert "\n" not in result
    assert "\t" not in result

#Texto muy corto no rompe la limpieza
def test_clean_text_very_short_text():
    result = clean_text("ok")
    assert isinstance(result, str)
    assert len(result) >= 0