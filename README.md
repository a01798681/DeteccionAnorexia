# Detector de Desórdenes Alimenticios

**Autores:** 
- Andrés Cabrera Alvarado - A01798681
- Andrea
- Pablo

**Proyecto:** Detección de señales asociadas a anorexia en textos de redes sociales  
**Estado actual:** Fase 3, con modelos clásicos, BETO, LLM y aplicación web React + FastAPI.

---

## 1. Descripción general

Este proyecto implementa una herramienta de análisis de texto para detectar señales asociadas a anorexia en publicaciones o mensajes escritos. La solución combina técnicas de Procesamiento de Lenguaje Natural, aprendizaje automático clásico, modelos basados en transformers y un modelo de lenguaje grande mediante Hugging Face.

El sistema permite:

- Clasificar textos individuales.
- Clasificar archivos CSV o Excel completos.
- Comparar predicciones entre varios modelos.
- Consultar métricas globales de desempeño por modelo.
- Agregar términos nuevos de jerga o vocabulario de redes sociales.
- Descargar resultados de clasificación en formato CSV.

> **Nota importante:** esta herramienta no debe utilizarse como diagnóstico médico, psicológico o clínico. Su objetivo es apoyar el análisis computacional de textos en redes sociales y comparar distintas estrategias de modelado.

---

## 2. Objetivo del proyecto

Desarrollar y evaluar una herramienta capaz de clasificar textos en dos categorías principales:

- `anorexia`
- `control`

Además, en las versiones más recientes se maneja una salida intermedia como `incierto` cuando la probabilidad cae en una zona ambigua definida por umbrales.

El objetivo técnico es comparar el mérito relativo de diferentes métodos:

1. Modelos clásicos con TF-IDF.
2. Modelos híbridos con TF-IDF + atributos manuales de riesgo.
3. RandomForest con reducción de dimensionalidad mediante SVD.
4. BETO + Logistic Regression.
5. LLM few-shot vía Hugging Face.
6. Métodos combinados BETO + LLM en cascada y ensamble.

---

## 3. Método recomendado actual

El método recomendado actual es:

```text
BETO + LLM ensemble
```

Este método combina:

- **BETO + Logistic Regression:** modelo transformer en español usado para obtener una probabilidad base de riesgo.
- **LLM vía Hugging Face:** modelo generativo usado para complementar la evaluación con un `risk_score` y una explicación breve.
- **Ensamble por puntaje:** combina ambos puntajes mediante una fórmula ponderada.

La fórmula usada conceptualmente es:

```text
score_final = alpha * score_BETO + beta * score_LLM
```

El sistema conserva también otros modelos para comparación y selección manual desde la interfaz.

---

## 4. Modelos disponibles

Los modelos registrados actualmente en la aplicación son:

| Key | Nombre en la interfaz | Familia | Descripción |
|---|---|---|---|
| `beto_llm_ensemble` | BETO + LLM ensemble | Avanzado | Método recomendado. Combina el score de BETO y el score del LLM. |
| `beto_llm_cascade` | BETO + LLM cascade | Avanzado | BETO decide si tiene alta confianza; el LLM se usa en casos ambiguos. |
| `beto_logreg` | BETO + Logistic Regression | Transformer | Embeddings de BETO con un clasificador de regresión logística. |
| `hybrid_logreg` | Logistic Regression híbrida | Clásico | TF-IDF + atributos manuales de riesgo. |
| `random_forest_svd` | RandomForest + SVD | Exploratorio | TF-IDF reducido con SVD y clasificador RandomForest. |

---

## 5. Resultados principales

Las métricas siguientes corresponden al conjunto de validación usado en el proyecto.

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| BETO + LLM ensemble | 0.9633 | 0.9808 | 0.9503 | 0.9653 | 0.9933 |
| BETO + LLM cascade | 0.9567 | 0.9625 | 0.9565 | 0.9595 | 0.9652 |
| BETO + Logistic Regression | 0.9400 | 0.9554 | 0.9317 | 0.9434 | 0.9853 |
| Logistic Regression híbrida | 0.9300 | 0.9321 | 0.9379 | 0.9350 | 0.9740 |
| RandomForest + SVD | 0.9067 | 0.8800 | 0.9565 | 0.9167 | 0.9655 |
| Logistic Regression optimizada | 0.9033 | 0.9125 | 0.9068 | 0.9097 | 0.9714 |
| Linear SVM | 0.9033 | 0.9177 | 0.9006 | 0.9091 | 0.9704 |
| spaCy + Logistic Regression híbrida | 0.9133 | 0.9042 | 0.9379 | 0.9207 | 0.9753 |

### Interpretación breve

- **BETO + LLM ensemble** obtuvo el mejor desempeño global y quedó como método recomendado.
- **BETO + LLM cascade** tuvo un recall ligeramente alto y funciona como alternativa avanzada.
- **BETO + Logistic Regression** es una opción fuerte y más práctica para clasificación masiva, porque evita llamar al LLM por cada fila.
- **Logistic Regression híbrida** se mantiene como baseline clásico sólido e interpretable.
- **RandomForest + SVD** alcanzó recall alto, aunque con menor precisión.

---

## 6. Requisitos

### Python

Se recomienda usar Python 3.10 o superior.

Dependencias principales:

- pandas
- openpyxl
- numpy
- scikit-learn
- joblib
- pytest
- streamlit
- python-dotenv
- huggingface_hub
- fastapi
- uvicorn
- python-multipart
- torch
- transformers
- spacy

Instalación base:

```bash
pip install -r requirements.txt
```

Si alguna dependencia del backend o BETO no está incluida en `requirements.txt`, instalar manualmente:

```bash
pip install fastapi uvicorn python-multipart torch transformers spacy
```

Para usar la variante con spaCy:

```bash
python -m spacy download es_core_news_sm
```

### Frontend

Se requiere Node.js y npm.

Desde la carpeta `frontend/`:

```bash
npm install
```

---

## 7. Variables de entorno

El archivo `.env` controla la integración con Hugging Face.

Ejemplo:

```env
HF_API_TOKEN=tu_token_de_hugging_face
HF_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
HF_USE_MOCK=false
```

### Campos

| Variable | Descripción |
|---|---|
| `HF_API_TOKEN` | Token de Hugging Face con permiso de lectura. |
| `HF_MODEL_ID` | Modelo LLM usado para la Fase 3. |
| `HF_USE_MOCK` | Si está en `true`, usa respuestas simuladas sin consumir créditos. |

Para pruebas sin consumo de créditos:

```env
HF_USE_MOCK=true
```

Para ejecutar el LLM real:

```env
HF_USE_MOCK=false
```

---

## 8. Ejecución del proyecto

### 8.1 Entrenamiento de modelos clásicos

Desde la raíz del proyecto:

```bash
python main.py
```

Este comando entrena y evalúa modelos clásicos como:

- Logistic Regression optimizada.
- Logistic Regression híbrida.
- Linear SVM.
- RandomForest + SVD.

Genera salidas en `results/`:

- modelos `.joblib`
- métricas `.json`
- predicciones de validación `.csv`
- falsos positivos y falsos negativos

---

### 8.2 Entrenamiento de BETO + Logistic Regression

```bash
python -m src.beto_train
```

Este proceso descarga/carga BETO, genera embeddings y entrena una regresión logística encima.

Salida principal:

```text
results/beto_logreg.joblib
results/beto_logreg_metrics.json
```

---

### 8.3 Evaluación del LLM

Prueba rápida:

```bash
python -m src.llm_smoke_test
```

Evaluación sobre validación:

```bash
python -m src.llm_evaluate
```

Evaluación limitada:

```bash
python -m src.llm_evaluate --limit 30
python -m src.llm_evaluate --limit 100
```

---

### 8.4 Evaluación BETO + LLM

Evaluar cascade:

```bash
python -m src.beto_llm_evaluate --method cascade
```

Evaluar ensemble:

```bash
python -m src.beto_llm_evaluate --method ensemble --alpha 0.7 --beta 0.3
```

Evaluar ambos:

```bash
python -m src.beto_llm_evaluate --method both
```

Para pruebas rápidas y menor consumo:

```bash
python -m src.beto_llm_evaluate --method both --limit 30
```

---

### 8.5 Entrenamiento con spaCy

```bash
python -m src.spacy_train
```

Este experimento aplica lematización y normalización con spaCy antes de entrenar el modelo híbrido.

---

### 8.6 Búsqueda de umbral

```bash
python -m src.threshold_search
```

Este script permite analizar si un cambio en el umbral de decisión mejora las métricas.

---

## 9. Ejecución de la aplicación web

La versión principal actual usa:

- **Backend:** FastAPI
- **Frontend:** React + Vite

### 9.1 Levantar backend

Desde la raíz del proyecto:

```bash
python -m uvicorn backend.main:app --reload
```

El backend queda disponible en:

```text
http://localhost:8000
```

Documentación automática de la API:

```text
http://localhost:8000/docs
```

---

### 9.2 Levantar frontend

En otra terminal:

```bash
cd frontend
npm run dev
```

El frontend queda disponible normalmente en:

```text
http://localhost:5173
```

---

### 9.3 Versión Streamlit legacy

La versión anterior en Streamlit se conserva como respaldo:

```bash
streamlit run app/app.py
```

---

## 10. Pruebas

Para ejecutar todas las pruebas:

```bash
pytest
```

Las pruebas cubren componentes como:

- carga de datos
- preprocesamiento
- features manuales
- evaluación
- predicción clásica
- predicción con BETO
- métodos BETO + LLM
- registro y runtime de modelos
- RandomForest
- pipeline end-to-end

---

## 11. Entradas y salidas

### Entradas

El sistema acepta:

- texto manual desde la interfaz
- archivos `.csv`
- archivos `.xlsx`
- datasets de entrenamiento en Excel
- términos personalizados de jerga

Columnas esperadas en dataset base:

```text
user_id
tweet_id
tweet_text
class
```

Clases esperadas:

```text
anorexia
control
```

### Salidas

El sistema produce:

- clase predicha
- probabilidad de anorexia
- nivel de confianza
- observaciones
- texto limpio
- métricas de evaluación
- matrices de confusión
- reportes de clasificación
- archivos CSV con predicciones
- análisis de falsos positivos y falsos negativos

---

## 12. Consideraciones de uso

### Sobre el uso de LLM

Los métodos que usan LLM pueden consumir créditos de Hugging Face, especialmente en clasificación por archivo. Para archivos grandes se recomienda usar:

```text
BETO + Logistic Regression
```

o

```text
Logistic Regression híbrida
```

El método `BETO + LLM ensemble` se recomienda principalmente para texto individual, comparación entre modelos o archivos pequeños.

### Sobre archivos grandes

Clasificar archivos grandes con LLM puede tardar bastante porque se procesa fila por fila. El sistema conserva caché para reducir llamadas repetidas, pero aun así el costo y el tiempo pueden ser altos.

### Sobre interpretación

Una predicción positiva no equivale a diagnóstico clínico. El resultado debe interpretarse como una señal computacional de riesgo textual.

---

## 13. Comandos rápidos

```bash
# Instalar dependencias Python
pip install -r requirements.txt

# Entrenar modelos clásicos
python main.py

# Entrenar BETO
python -m src.beto_train

# Evaluar LLM
python -m src.llm_evaluate

# Evaluar BETO + LLM ensemble y cascade
python -m src.beto_llm_evaluate --method both

# Ejecutar backend
python -m uvicorn backend.main:app --reload

# Ejecutar frontend
cd frontend
npm install
npm run dev

# Ejecutar pruebas
pytest
```

---

## 14. Estado actual

El proyecto cuenta actualmente con:

- modelos clásicos entrenados
- modelo BETO entrenado
- integración con LLM vía Hugging Face
- métodos combinados BETO + LLM
- backend FastAPI
- frontend React + Vite
- edición de términos personalizados
- clasificación individual y por archivo
- comparación visual entre modelos
- métricas globales visibles desde el frontend
- pruebas unitarias y de integración