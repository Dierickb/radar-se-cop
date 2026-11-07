"""Pasos 2 y 3 – Limpieza, normalización y preprocesamiento lingüístico.

Dos variantes de texto, para poder medir el efecto de cada decisión:
- `normalizar`: minúsculas, sin tildes, sin números ni códigos.
- `limpiar`: además quita las fórmulas de modalidad contractual ("prestación de
  servicios", "apoyo a la gestión", "por sus propios medios"...). Estas frases
  describen CÓMO se contrató a una persona natural, no QUÉ servicio recibe la
  entidad, y en SECOP están asociadas a la familia 8011 (atajo que hereda el sesgo).
- `lematizar`: spaCy es_core_news_sm (lemas, sin stopwords).
"""
import re
import unicodedata
from functools import lru_cache

# Fórmulas de modalidad contractual (medido en la muestra: P(familia 8011 | frase) entre 0,75 y 0,97)
FORMULAS = [
    r"prestaci[oó]n de (los )?servicios? (profesionales|t[eé]cnicos|asistenciales|personales)?( y de apoyo a la gesti[oó]n| de apoyo a la gesti[oó]n)?",
    r"prestar (los |sus )?servicios? (profesionales|t[eé]cnicos|asistenciales|personales)?( y de apoyo a la gesti[oó]n| de apoyo a la gesti[oó]n)?",
    r"servicios? (profesionales|t[eé]cnicos|asistenciales) (y )?de apoyo a la gesti[oó]n",
    r"servicios? de apoyo a la gesti[oó]n",
    r"apoyo a la gesti[oó]n",
    r"por sus propios medios",
    r"con (plena|total) autonom[ií]a( t[eé]cnica)?( y administrativa)?",
    r"contrato de prestaci[oó]n de servicios",
    # restos sueltos cuando la fórmula viene en otro orden
    r"servicios? (profesionales|t[eé]cnicos|asistenciales) (especializados )?",
    r"\bprestar\b",
    r"\bprestaci[oó]n\b",
]
_FORMULAS_RX = re.compile("|".join(f"(?:{f})" for f in FORMULAS))
_CODIGO_RX = re.compile(r"\b[a-z]*\d[\w-]*\b")          # códigos tipo SHF690, 11643-, CO1.PCCNTR...
_NO_LETRA_RX = re.compile(r"[^a-zñ ]+")


def _sin_tildes(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))  # la ñ queda como n


def normalizar(texto: str) -> str:
    s = str(texto).lower()
    s = _CODIGO_RX.sub(" ", s)
    s = _sin_tildes(s)
    s = _NO_LETRA_RX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def quitar_formulas(texto: str) -> str:
    """Se aplica sobre el texto en minúsculas y CON tildes (las regex las toleran)."""
    return _FORMULAS_RX.sub(" ", str(texto).lower())


def limpiar(texto: str) -> str:
    return normalizar(quitar_formulas(texto))


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("es_core_news_sm", disable=["parser", "ner"])


def lematizar(textos, batch_size: int = 1000, n_process: int = 1) -> list[str]:
    """Lemas sin stopwords ni palabras de una letra. Entrada ya limpia."""
    out = []
    for doc in _nlp().pipe(textos, batch_size=batch_size, n_process=n_process):
        out.append(" ".join(_sin_tildes(t.lemma_.lower()) for t in doc
                            if not t.is_stop and t.is_alpha and len(t) > 1))
    return out


def preparar(texto: str) -> str:
    """Pipeline completo para un texto suelto (lo usa la app)."""
    return lematizar([limpiar(texto)])[0]
