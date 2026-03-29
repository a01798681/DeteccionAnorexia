from src.preprocessing import fix_mojibake, clean_text


def test_fix_mojibake():
    text = "mÃ¡s frÃ­o"
    fixed = fix_mojibake(text)
    assert "más" in fixed or "frío" in fixed


def test_clean_text_removes_url():
    text = "hola mira esto https://ejemplo.com"
    cleaned = clean_text(text)
    assert "https" not in cleaned


def test_clean_text_removes_mentions():
    text = "@usuario hola cómo estás"
    cleaned = clean_text(text)
    assert "@usuario" not in cleaned


def test_clean_text_keeps_hashtags():
    text = "quiero bajar de peso #thinspo"
    cleaned = clean_text(text)
    assert "#thinspo" in cleaned