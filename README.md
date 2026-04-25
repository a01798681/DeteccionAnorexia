# DeteccionAnorexia
## Descripción
Este proyecto implementa una herramienta para detectar señales asociadas a anorexia en textos de redes sociales mediante técnicas de Procesamiento de Lenguaje Natural (PLN) e Inteligencia Artificial (IA).

La solución incluye:
- Preprocesamiento de texto,
- Extracción de atributos mediante TF-IDF y features manuales,
- Entrenamiento y optimización de modelos de clasificación,
- Evaluación mediante AUC,
- Interfaz en Streamlit para clasificación individual y por archivo.

## Objetivo
Desarrollar una herramienta capaz de clasificar textos en dos categorías:
- **anorexia**
- **control**

y comparar distintas estrategias de modelado para seleccionar la mejor solución.

## Tecnologías utilizadas
Python
pandas
scikit-learn
openpyxl
joblib
pytest
Streamlit

## Instala las dependencias con:
```bash
pip install -r requirements.txt
```
## Ejecución del entrenamiento

### Para entrenar y evaluar los modelos:
```bash
python main.py
```
Esto genera:
- modelos entrenados en results/
- métricas en formato .json
- archivos de análisis de errores (false_positives, false_negatives, etc.)
## Ejecución de la aplicación

Para abrir la interfaz web en Streamlit:
```bash
streamlit run app/app.py
```
## Pruebas

Para ejecutar las pruebas unitarias:
```bash
pytest
```
## Modelo final seleccionado

El modelo principal del proyecto es:
```bash
logistic_regression_hybrid_v1.joblib
```
### Este modelo combina:
representación TF-IDF
atributos manuales de riesgo
Resultados principales

## Resultados del modelo final:
- **ROC-AUC: 0.9546**
- **Accuracy: 0.8900**

## Comparación:
- Logistic Regression optimizada: AUC = 0.9469
- Linear SVM: AUC = 0.9440
- Logistic Regression híbrida v1: AUC = 0.9546

## Entradas y salidas
### Entradas
archivo de entrenamiento en Excel texto ingresado manualmente en la app archivo .csv o .xlsx cargado por el usuario

### Salidas
clase predicha, probabilidad estimada, observaciones, métricas de evaluación, archivos de resultados