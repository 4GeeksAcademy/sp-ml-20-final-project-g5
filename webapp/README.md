# WebApp (Streamlit) – Predicción de ETA

Esta carpeta contiene una app *Streamlit* para predecir el **tiempo estimado (días)** usando un modelo entrenado(`.pkl`)

## Qué hace la app

1. El usuario introduce los datos del pedido (importe, envío, cuotas, categoría, hora/día, etc.).
2. El usuario introduce el **prefijo del código postal**.
3. Si el CP existe en el diccionario `lookup_zip.csv`, la app **autocompleta** ciudad/estado/coordenadas.
4. El modelo devuelve la predicción: **ETA estimado en días**.

## Requisitos
    - Python **3.10+**
- Paquetes principales:
  - `streamlit`
  - `pandas`
  - `scikit-learn`
- (El resto de dependencias vienen en el `requirements.txt` del repositorio, si aplica)

## Estructura esperada

La app espera estos archivos en el repositorio:

- `webapp/app.py` → aplicación Streamlit
- `models/random_forest_delivery_time.pkl` → modelo entrenado (Pipeline)
- `data/processed/df_model.csv` → dataset de referencia (opcional, para listas desplegables)
- `data/processed/lookup_zip.csv` → diccionario CP → ciudad/estado/lat/lng
- `webapp/assets/hero.jpeg` → imagen de cabecera

## Ejecutar en local / codespaces
Desde la raíz del repositorio

```bash
streamlit run webapp/app.py ```

## Nota (MVP)

Si se reentrena el modelo con nuevas variables, hay que actualizar el formulario y `features` en `webapp/app.py`.