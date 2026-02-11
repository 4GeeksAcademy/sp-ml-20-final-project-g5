# 📦 ¿Cuándo llegará mi pedido?
## Predicción del tiempo de entrega en e-commerce con Machine Learning

---

## 🧠 Nombre del proyecto
¿CUÁNDO LLEGARÁ MI PEDIDO?  
Predicción del tiempo de entrega en e-commerce con Machine Learning

---

## 🎯 Objetivo del proyecto
El objetivo de este proyecto es predecir el tiempo de entrega de un pedido (en días) en un entorno de comercio electrónico utilizando técnicas de Machine Learning. A partir de información del pedido, del cliente, del vendedor y de su localización geográfica, se busca estimar de forma realista cuánto tardará un pedido en ser entregado.

---

## 📊 ¿Cómo se obtuvo acceso a los datos?
Los datos utilizados provienen del dataset público de Olist, una plataforma brasileña de comercio electrónico, disponible en Kaggle. Este conjunto de datos contiene información histórica sobre pedidos, clientes, productos, vendedores, pagos, reseñas y geolocalización.

---

## ⚠️ Dificultades en el proceso
Durante el desarrollo del proyecto se presentaron varias dificultades relevantes:

- Integración de múltiples tablas mediante procesos de merge.
- Detección y eliminación de registros duplicados.
- Presencia de valores nulos y outliers en el tiempo de entrega.
- Riesgo de data leakage al trabajar con variables relacionadas con el estado del pedido. (Reseñas o Estado del pedido)
- Alta variabilidad del tiempo de entrega debido a factores externos no incluidos en los datos.

Estas dificultades se abordaron mediante limpieza de datos, validaciones y una cuidadosa selección de variables.

---

## 📈 Métricas y modelos
Se entrenaron y compararon distintos modelos de regresión para predecir el tiempo de entrega:

- Regresión Lineal
- Ridge
- Lasso
- Random Forest
- Gradient Boosting

Las métricas utilizadas para la evaluación fueron:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coeficiente de determinación)

El modelo Gradient Boosting obtuvo el mejor rendimiento global, aunque con mejoras moderadas frente a Random Forest, reflejando la complejidad del problema logístico.

---

## 🚀 Despliegue en Streamlit
El modelo final fue desplegado mediante una aplicación web desarrollada en Streamlit, que permite al usuario introducir información básica del pedido y obtener una estimación del tiempo de entrega.

La aplicación incluye:
- Uso del pipeline completo del modelo entrenado.
- Autocompletado de coordenadas a partir del código postal.
- Validación de entradas y manejo de errores.

Este despliegue simula un entorno real de uso del modelo en producción.

---

## 🔮 Posibles mejoras
De cara a futuras iteraciones del proyecto, se identifican varias líneas de mejora:

- Incorporar información adicional sobre transportistas, rutas o condiciones climáticas.
- Calcular distancias más precisas considerando redes de transporte reales.
- Probar modelos de boosting más avanzados como XGBoost o LightGBM.
- Realizar un ajuste más exhaustivo de hiperparámetros.
- Ampliar el despliegue con funcionalidades de seguimiento y visualización del estado del pedido.

---

### Nota sobre la variable de distancia geográfica

Durante el desarrollo del proyecto se exploró la incorporación de la variable `distance_km`, que representa la distancia geográfica aproximada entre el cliente y el vendedor. Al incluir esta variable en el proceso de modelado, se observó una mejora significativa en el rendimiento del modelo, alcanzando valores de R² superiores a los obtenidos en la versión final desplegada.

Sin embargo, por limitaciones de tiempo y para garantizar la estabilidad del despliegue en Streamlit, esta variable no fue incluida en la versión final de la aplicación, ya que su incorporación requería modificaciones adicionales en el flujo de entrada de datos y en el pipeline del modelo. Aun así, estos resultados evidencian la relevancia de la distancia geográfica como uno de los principales factores explicativos del tiempo de entrega.
