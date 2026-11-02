### Practica-Mineria-de-Textos
Práctica final de la asignatura de Minería de Textos

# Clasificación de textos en Esperanto: Deportes

Práctica de la asignatura **Minería de Textos** (5º Ingeniería Informática y ADE, CUNEF Universidad, curso 2026/2027).

**Participantes** Alberto Sainz, José Emparanza e Inés Devis

---

## Objetivo del proyecto

El Esperanto es la lengua artificial más hablada del mundo, pero casi no tiene herramientas de Procesamiento de Lenguaje Natural (NLP). El objetivo de este proyecto es construir, de principio a fin, un sistema capaz de **clasificar textos en Esperanto según su temática**.

Hemos elegido tres categorías del ámbito deportivo:

| Clase | Deporte |
|---|---|
| `futbol` | Fútbol |
| `baloncesto` | Baloncesto |
| `tenis` | Tenis |

Son categorías **semánticamente cercanas**: comparten mucho vocabulario (*ludanto*, *teamo*, *matĉo*, *venki*, *ĉampionado*…), así que el problema de clasificación no es trivial.

El proyecto se divide en tres fases:

1. **Fase 1 – Dataset:** recopilación automática de textos en Esperanto desde Wikipedia.
2. **Fase 2 – NLP:** desarrollo de un modelo de NLP propio para Esperanto, basado en reglas y con un API similar al de spaCy.
3. **Fase 3 – Clasificación:** vectorización del dataset, entrenamiento de un clasificador y clasificación de páginas web reales *(en desarrollo)*.

---

## Contenido del repositorio

| Fichero | Descripción |
|---|---|
| `Deportes.ipynb` | Cuaderno Jupyter con todo el código del proyecto (ejecutable en Google Colab) |
| `dataset_esperanto_deportes2.csv` | Dataset generado en la Fase 1 |
| `README.md` | Este documento |

---

## Fase 1: Dataset

### Fuente de datos

Los textos se obtienen de la **API de Wikipedia en Esperanto** (`eo.wikipedia.org`). Con el parámetro `prop=extracts` y `explaintext=True` obtenemos el texto plano del artículo, ya sin HTML, enlaces ni referencias. Con `redirects=1` la API sigue las redirecciones, así que podemos pedir un artículo aunque su título haya cambiado.

### Proceso

1. **Artículos semilla:** para cada clase definimos a mano una lista de unos 40 artículos representativos (el deporte, sus competiciones, equipos, selecciones y jugadores más conocidos).
2. **Descarga:** se pide cada artículo a la API. Hay hasta 3 reintentos por artículo y una pausa de 0,35 s entre peticiones para no saturar el servidor. Si la API devuelve un error 429 (demasiadas peticiones), se espera más tiempo.
3. **Artículos no encontrados:** se registran y se muestran por pantalla, porque algunos títulos no existen en la Wikipedia en Esperanto.
4. **Completar con categorías:** si una clase no llega a 350 párrafos, se completa con artículos de su categoría de Wikipedia (`Kategorio:Futbalo`, `Kategorio:Basketbalo`, `Kategorio:Teniso`) y de sus subcategorías, un nivel de profundidad.
5. **Equilibrado:** si una clase supera los 450 párrafos, se recorta con una muestra aleatoria de semilla fija.
6. **Guardado:** el resultado se mezcla aleatoriamente y se guarda en CSV.

### Limpieza de los párrafos

Cada artículo se divide en párrafos (por saltos de línea) y a cada uno se le aplica:

- Se descartan los títulos de sección (líneas del tipo `== Historio ==`).
- Se sustituye `;` por `,` para respetar el separador del CSV.
- Se normalizan los espacios en blanco.
- Se descartan los párrafos de menos de **60 caracteres**, porque suelen ser restos de listas o frases sin contenido.
- Se descartan los párrafos en los que más del **25 % de los caracteres son dígitos** (listas de resultados, años, estadísticas).

También se eliminan los duplicados: dentro de cada clase se quedan una sola vez, y si un mismo párrafo aparece en dos clases distintas se elimina de ambas, porque es ambiguo. Además, un mismo artículo no se usa nunca en dos clases.

### Contrato de datos

El dataset cumple el contrato pedido en el enunciado:

- Formato **CSV**, codificación **UTF-8** y separador `;`.
- Dos columnas con cabecera `text;class`.
- Granularidad de **párrafo**.
- Ningún párrafo contiene el carácter `;`.

### Resultado

| Clase | Párrafos |
|---|---|
| `futbol` | 382 |
| `tenis` | 264 |
| `baloncesto` | 143 |
| **Total** | **789** |

### Medición del ruido

Cada párrafo hereda la etiqueta del artículo del que sale, y esto introduce **ruido**: muchos párrafos son genéricos y no tienen nada específico de su deporte (por ejemplo, *"Ĝi estis fondita en 1950"*).

Para estimarlo, comprobamos si cada párrafo contiene alguna **raíz clave** de su deporte. Buscamos raíces y no palabras completas para cubrir todas las terminaciones (plural, acusativo, derivados…), por ejemplo `futbal` encuentra *futbalo*, *futbalisto* y *futbalan*. Si un párrafo no contiene ninguna, lo consideramos genérico.

| Clase | Párrafos genéricos (ruido estimado) |
|---|---|
| `baloncesto` | 36,4 % |
| `futbol` | 29,1 % |
| `tenis` | 26,9 % |

Muchos de estos párrafos describen reglas del juego, estadísticas o biografías sin nombrar el deporte, así que sí tienen información útil para el clasificador. Es una **medida aproximada**: sirve para tener una idea del orden de magnitud del ruido, no para limpiarlo automáticamente.

### Ejecución asíncrona

La variable `GENERAR_DATASET` permite elegir entre dos modos:

- `True`: se descarga y genera el dataset desde Wikipedia.
- `False`: se carga el CSV ya generado desde este repositorio de GitHub.

Así los resultados son **reproducibles** aunque Wikipedia cambie con el tiempo, y el cuaderno sigue funcionando aunque la API no esté disponible.

---

## Fase 2: Modelo NLP para Esperanto

Hemos desarrollado desde cero un modelo de NLP para Esperanto **basado en reglas**, sin usar librerías de NLP existentes (spaCy, NLTK, Stanza…). Aprovechamos que el Esperanto es una lengua con una gramática muy regular: la categoría gramatical de una palabra se puede deducir casi siempre por su terminación.

### Uso (API similar a spaCy)

```python
nlp = EsperantoNLP()
doc = nlp("La famaj futbalistoj rapide ludis en la granda hispana stadiono.")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.is_stop, token.is_punct)
```

### Clases

| Clase | Descripción |
|---|---|
| `EsperantoNLP` | El pipeline. Al llamarlo con un texto devuelve un `Doc`. |
| `Doc` | Secuencia de tokens. Se puede recorrer, tiene `len()` y acceso por índice (`doc[0]`). |
| `Token` | Cada palabra o signo, con sus atributos: `text`, `lemma` / `lemma_`, `pos` / `pos_`, `is_stop`, `is_punct`, `is_alpha`, `lower_` e `i` (posición en el `Doc`). |

### Componentes

**1. Tokenizador.** Usa una expresión regular que separa palabras (incluidas las letras propias del Esperanto: ĉ, ĝ, ĥ, ĵ, ŝ, ŭ) y signos de puntuación.

**2. Etiquetador morfológico (POS-tagging) y lematizador.** Las reglas se aplican en un **orden concreto**. Primero se buscan las palabras de las listas cerradas, que no siguen las terminaciones regulares, y después se aplican las reglas por terminación. El orden es importante: sin él, palabras como *tri* (tres) o *plu* (más) se etiquetarían como verbos por terminar en *-i* / *-u*.

| Orden | Regla | Etiqueta | Ejemplo → lema |
|---|---|---|---|
| 1 | Signo de puntuación | `PUNCT` | `.` |
| 2 | Pronombres personales | `PRON` | *ĝin* → *ĝi* |
| 3 | Correlativos (tabla *ti-/ki-/i-/ĉi-/neni-*) | `PRON` | *kiun* → *kiu* |
| 4 | Preposiciones | `ADP` | *kontraŭ* |
| 5 | Conjunciones | `CCONJ` | *kaj* |
| 6 | Números y numerales | `NUM` | *2022*, *tri* |
| 7 | Artículo | `DET` | *la* |
| 8 | Partículas | `PART` | *ĉu*, *ne* |
| 9 | Adverbios primitivos | `ADV` | *ankaŭ*, *plu* |
| 10 | Mayúscula y no al inicio de frase | `PROPN` | *Messi* |
| 11 | Terminación verbal *-as, -is, -os, -us, -u, -i* | `VERB` | *ludis* → *ludi* |
| 12 | Terminación *-o, -oj, -on, -ojn* | `NOUN` | *futbalistoj* → *futbalisto* |
| 13 | Terminación *-a, -aj, -an, -ajn* | `ADJ` | *flavan* → *flava* |
| 14 | Terminación *-e* | `ADV` | *rapide* |
| 15 | Ninguna de las anteriores | `X` | |

Para lematizar se aplica la forma de diccionario de cada categoría: los verbos pasan al infinitivo (*-i*), los sustantivos y adjetivos pierden el plural (*-j*) y el acusativo (*-n*), y los pronombres y correlativos pierden el acusativo.

**3. Gestión de stop-words.** Las stop-words se cargan del recurso [`vortoj.json`](https://github.com/jparisu/nlp-esperantilo/blob/main/resources/esperanto/vortoj.json) del repositorio de guía de la asignatura. Se cogen las 122 palabras marcadas como `ignorinda`. Si el fichero no se puede descargar, se usa un conjunto base propio (artículo, pronombres, preposiciones, conjunciones y algunos adverbios frecuentes). Un token es stop-word si lo es su forma original o su lema.

### Limitaciones conocidas

- Un **nombre propio al principio de una frase** no se puede distinguir de una palabra normal, así que se le aplican las reglas por terminación.
- Los nombres propios **conservan el acusativo** (*Mondpokalon*). Quitar la *-n* final rompería nombres extranjeros como *Johnson* o *Wilson*.
- El lematizador trabaja a **nivel de palabra**, no de raíz: *futbalisto* (futbolista) y *futbalo* (fútbol) tienen lemas distintos aunque compartan la raíz *futbal-*.
- Las palabras extranjeras en minúscula que terminan como una palabra en Esperanto pueden recibir una etiqueta incorrecta.

---

## Fase 3: Clasificación

*En desarrollo.* En esta fase usaremos el modelo NLP de la Fase 2 para vectorizar el dataset y entrenar un clasificador. Lo evaluaremos con *accuracy*, F1 macro y matriz de confusión, y lo probaremos con páginas web reales en Esperanto.

---

## Cómo ejecutarlo

1. Abrir `Deportes.ipynb` en [Google Colab](https://colab.research.google.com/).
2. Ejecutar todas las celdas en orden (*Entorno de ejecución → Ejecutar todas*).
3. Para no descargar el dataset de nuevo, poner `GENERAR_DATASET = False` en la primera celda de código.

**Dependencias:** `requests`, `pandas` (ya instaladas en Colab). No se necesita GPU.

---

## Referencias

- Repositorio de guía de la asignatura: [jparisu/nlp-esperantilo](https://github.com/jparisu/nlp-esperantilo)
- Guía de gramática del Esperanto: [nlp-esperantilo/guide/esperanto](https://jparisu.github.io/nlp-esperantilo/es/guide/esperanto/)
- API de Wikipedia en Esperanto: [eo.wikipedia.org/w/api.php](https://eo.wikipedia.org/w/api.php)
