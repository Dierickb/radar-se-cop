# Radar SECOP

**Clasificación explicable de contratos públicos colombianos en familias UNSPSC, para que las pymes encuentren procesos de SECOP II y para revisar la calidad de los códigos declarados.**

Proyecto final de Procesamiento de Lenguaje Natural, Pontificia Universidad Javeriana.
Autores: David Moreno y Dierick Brochero.

## Problema y usuario

Toda compra pública en Colombia se publica en SECOP II con un código del clasificador UNSPSC (estándar de la ONU adoptado por Colombia Compra Eficiente). Para una pyme, ese código es la puerta de entrada: si no sabe en qué familia cae lo que vende, no encuentra los procesos que le sirven. El catálogo tiene 420 familias y está escrito en lenguaje de catálogo, no en el de un empresario.

Radar SECOP recibe la descripción del negocio **en palabras de la pyme** y devuelve:

1. las 3 familias UNSPSC más probables, con la explicación de por qué (contribución exacta de cada palabra y contratos reales parecidos);
2. los procesos publicados en SECOP II que mejor le encajan (búsqueda híbrida semántica + léxica + familia).

Un segundo usuario, veedurías y equipos de control interno, usa el mismo modelo para marcar contratos cuyo código declarado no coincide con lo que dice su objeto. Esos contratos quedan **para revisar**: una discrepancia no es un error ni una irregularidad.

**Tarea de PLN:** clasificación de texto multiclase (175 familias con datos suficientes), más recuperación de información y explicabilidad.

## Hallazgo central: las etiquetas de SECOP están sesgadas

La familia 8011 (Servicios de recursos humanos) se usa como cajón de sastre para casi todo contrato de prestación de servicios. La guía oficial de codificación (G-CBS-02, 2026, Caso 10) dice que un servicio profesional debe codificarse según el servicio que recibe la entidad: un abogado va en 8012, Servicios legales. En los datos, en cambio:

- en el conjunto de test, el 83 % de los contratos con "abogado" están en 8011 y solo el 9 % en 8012;
- en el análisis exploratorio, frases de trámite como "por sus propios medios" predicen 8011 con P = 0,97 (la tasa base es 0,68), y 188 de 1.139 entidades ponen el 95 % o más de sus contratos en el segmento 80.

Esto cambia el diseño del proyecto:

- Las métricas contra SECOP miden **acuerdo con SECOP, no exactitud**.
- Para la evaluación real se anota a mano un **gold set** de 300 contratos con una guía basada en el Caso 10. Lo anota una sola persona; la confiabilidad se mide re-anotando a ciegas 60 contratos una semana después (kappa de Cohen intra-anotador).
- Las fórmulas de contratación se quitan del texto para que el modelo no aprenda el atajo hacia 8011.

## Pipeline

```
 SECOP II (datos.gov.co, API SODA)        Catálogo UNSPSC v14 en español (Colombia Compra Eficiente)
          │                                              │
 01 descarga ─────────────────────────────────────────────┘
          │  contratos 2025–2026 (objeto + código declarado) · procesos publicados/abiertos
 02 preparación de texto
          │  normalizar (minúsculas, sin tildes ni códigos) → quitar fórmulas de contratación
          │  → lematizar con spaCy es_core_news_sm, sin stopwords
          │  variante A = solo normalizado · variante B = sin fórmulas + lematizado (la de la app)
 03 etiquetas + modelo
          │  familia = 4 primeros dígitos del código · familias con ≥ 30 ejemplos (175)
          │  split 80/20 agrupado por texto · tope de 3.000 por familia
          │  el gold set se excluye del entrenamiento (por id y por texto idéntico)
          │  TF-IDF (1–2 gramas, 200k términos) + regresión logística uno-contra-resto (SGD, log-loss)
 04 explicabilidad: global por familia · local exacta (tf-idf × coeficiente) · contrafactuales
          │  · vecinos más cercanos · fidelidad ERASER contra referencia al azar · calibración del umbral
 05 sesgo por entidad: split por entidad (entidades nunca vistas)
 06 discrepancias: código declarado ≠ predicción con confianza ≥ 0,70 → "para revisar"
 07 índice de procesos: texto + familia (inferida si viene sin código) + embeddings model2vec
 08 gold set: kappa intra-anotador · error de SECOP por estrato · Claude vs. gold · modelos vs. gold
          │  · plausibilidad de la explicación (palabras del modelo vs. palabras clave humanas)
 09 variante con datos corregidos por Claude (solo si kappa Claude–gold ≥ 0,61)
 10 evaluación de la búsqueda: consultas escritas por personas, juicios a ciegas, P@5 y nDCG@10
          │
 app/ Streamlit: "Para pymes" · "Revisar un contrato" · "Cómo funciona"
```

### Ejemplo real de punta a punta

| Paso | Valor |
|---|---|
| Objeto original | "Prestación de servicios profesionales como abogado para el acompañamiento jurídico de la personería municipal" |
| Código declarado | 80111600 → familia **8011** Servicios de recursos humanos |
| Sin fórmulas + lematizado | `abogado acompanamiento juridico personeria municipal` |
| Predicción (variante B) | **8012** Servicios legales, P = 0,73 |
| Explicación | "jurídico" (+2,77) y "abogado" (+1,78) son los términos que más empujan hacia 8012 |
| Contrafactual | sin "abogado jurídico", el modelo cae a 8011 con P = 0,16: sin el oficio, solo queda el cajón de sastre |
| App | se marca **para revisar** (8011 declarado ≠ 8012 sugerido con confianza ≥ 0,70) |

## Resultados actuales (contra etiquetas de SECOP)

Familia (175 clases). El test natural tiene la distribución real (63 % de 8011); el balanceado tiene como máximo 300 por familia.

| Variante | Test | Accuracy | Macro-F1 | Top-3 | ECE |
|---|---|---|---|---|---|
| Siempre la clase mayoritaria | natural | 0,629 | 0,004 | – | – |
| A: normalizado (con fórmulas) | natural | 0,413 | 0,307 | 0,712 | 0,085 |
| B: sin fórmulas + lematizado | natural | 0,397 | 0,302 | 0,704 | 0,096 |
| A: normalizado (con fórmulas) | balanceado | 0,559 | 0,408 | 0,751 | 0,091 |
| B: sin fórmulas + lematizado | balanceado | 0,559 | 0,407 | 0,759 | 0,074 |
| B, entidades nunca vistas | balanceado | 0,428 | 0,293 | 0,649 | 0,040 |

Cómo leer la tabla:

- La accuracy de "siempre 8011" es alta por el desbalance, pero su macro-F1 es casi cero. Por eso se reporta macro-F1 y top-3.
- Quitar las fórmulas (B) casi no cambia el acuerdo con SECOP. Lo que cambia es **qué aprende el modelo**: en A, entre los términos más fuertes de 8011 están "servicios profesionales" (peso 3,07) y "prestar servicios" (2,96). En B esas frases desaparecen, aunque aparecen atajos nuevos de oficina y lugar ("secretaría", "gobernación"); se documenta como limitación.
- Con entidades nunca vistas, el macro-F1 balanceado baja de 0,41 a 0,29: parte de lo que el modelo aprende son costumbres de cada entidad.
- **Oficios:** en test, SECOP pone al 83 % de los abogados en 8011; el modelo B predice 8012 para el 59 %. En enfermería, SECOP pone el 51 % en 8510 y el modelo B el 78 %.
- **Explicabilidad (ERASER, 500 contratos de test):** quitar las 3 palabras que el modelo señala baja la probabilidad de la clase 0,29, frente a 0,09 si se quitan 3 al azar (comprehensiveness). Las palabras que el modelo señala sí son las que usa.
- **Umbral de discrepancia:** con confianza ≥ 0,9 el modelo coincide con SECOP en el 95 % de los casos, pero cubre solo el 1,2 % de los contratos. Con ≥ 0,7, cubre el 14 % y coincide en el 68 %. Con el umbral de 0,70 se marcan 1.640 de 36.844 contratos de test (4,5 %), y el 80 % de ellos están declarados en 8011.

La evaluación contra el gold set humano (paso 08) está pendiente de que termine la anotación.

## Instalación

Probado con Python 3.13, scikit-learn 1.9 y spaCy 3.8; se recomienda Python 3.13 para obtener las mismas cifras.

```bash
git clone https://github.com/davidm094/radar-secop && cd radar-secop
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt   # versiones exactas usadas (o requirements.txt para las últimas)
pip install --no-deps https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.7.0/es_core_news_sm-3.7.0-py3-none-any.whl
```

Se fija el modelo de spaCy 3.7.0 porque es con el que se calcularon las cifras (`--no-deps` evita que pip baje spaCy a 3.7; con spaCy 3.8 sale un aviso de versión, pero funciona).

## Cómo correrlo

**Para reproducir exactamente las cifras del documento** (recomendado), use la foto de datos del 14-oct-2026
que trae el repo y sáltese los pasos 01 y 02 (SECOP cambia a diario, así que descargar de nuevo da otras cifras):

```bash
python scripts/00_usar_datos_congelados.py  # copia data/congelado/ a data/processed/ y data/raw/
python scripts/03_entrenar_evaluar.py       # ~5 min; test balanceado B: macro-F1 0,407, top-3 0,759
python scripts/07_indexar_procesos.py       # ~20 min
streamlit run app/app.py
```

Para descargar datos nuevos desde cero:

```bash
python scripts/01_descargar.py              # ~300k contratos 2025+, procesos abiertos y catálogo (20–40 min)
python scripts/02_preparar.py               # limpieza y lematización
python scripts/03_entrenar_evaluar.py       # modelos A y B + métricas + auditoría del atajo
python scripts/04_explicabilidad.py
python scripts/05_particion_entidad.py
python scripts/06_discrepancias.py
python scripts/07_indexar_procesos.py       # índice para la búsqueda
streamlit run app/app.py                    # demo en http://localhost:8501
```

Con el gold set anotado y, opcionalmente, con Claude:

```bash
python scripts/08_acuerdo_gold.py --repeticion                  # al terminar los 300: 60 ítems para re-anotar a ciegas
python scripts/08_acuerdo_gold.py --desacuerdos                 # kappa intra-anotador + inconsistencias, sin abrir la clave
python scripts/08_acuerdo_gold.py                               # evaluación completa contra el gold set
python scripts/11_revisar_discrepancias.py evaluar              # precisión de la alerta (revisión humana)
python llm/reetiquetar.py --entrada data/gold/gold_set_plantilla.xlsx --salida data/llm/gold_claude.jsonl
python scripts/09_entrenar_con_llm.py                           # solo corre si kappa Claude–gold ≥ 0,61
python scripts/10_evaluar_busqueda.py pool                      # luego marcar relevancia a mano
python scripts/10_evaluar_busqueda.py evaluar
```

Los modelos y el índice no se suben al repositorio porque pesan más de 100 MB; se regeneran con los pasos 03 y 07. Los resultados (métricas, explicaciones, discrepancias) quedan en `reports/`.

## Estructura

```
src/radar/      config · data (descarga SODA y catálogo) · text (limpieza y lematización) · labels (niveles,
                split, balanceo, exclusión del gold) · model · evaluate (métricas y ECE) · explain · search
scripts/        00–11, un paso del pipeline cada uno (00 = datos congelados, 11 = revisión de discrepancias)
app/app.py      demo en Streamlit
llm/            re-etiquetado y aumento con Claude Code (ver llm/README.md)
notebooks/      beto_colab.ipynb: BETO afinado en Colab (GPU) con el mismo split que el modelo lineal
data/beto/      split congelado (train balanceado + test) para que la comparación con BETO sea justa
data/gold/      ids del gold set y planillas de anotación
reports/        métricas y salidas de cada paso
tests/          simulador de Claude para probar llm/ sin gastar cuota
```

## Uso de LLM

Claude se usa para dos tareas de preparación de datos y **nunca** en la app ni en la evaluación:

1. re-etiquetar contratos con la misma guía que los anotadores humanos, validado primero contra el gold set;
2. generar descripciones de negocio estilo pyme para entrenamiento.

Detalle, comandos y controles en [`llm/README.md`](llm/README.md).

## Limitaciones conocidas

- Las etiquetas de entrenamiento son las de SECOP, con el sesgo descrito; el gold set es pequeño (300).
- El lematizador de spaCy comete errores en algunas palabras ("área" → "areo"); se usa el modelo `es_core_news_sm` 3.7 con spaCy 3.8.
- Al quitar las fórmulas aparecen atajos de oficina y municipio; con entidades nuevas el desempeño cae.
- El catálogo en español disponible es v14; la guía 2026 cita UNSPSC v26, publicado solo en inglés.
- Los procesos "abiertos" son una foto del día de descarga.

## Datos y licencias

- SECOP II – Contratos Electrónicos (`jbjy-vk9h`) y Procesos de Contratación (`p6dx-8zbt`), datos.gov.co, licencia CC BY-SA 4.0.
- Clasificador de Bienes y Servicios v14 en español, Colombia Compra Eficiente.
- Guía de Codificación de Bienes y Servicios G-CBS-02 (2026), Colombia Compra Eficiente.
- Embeddings: `minishlab/potion-multilingual-128M` (model2vec).
