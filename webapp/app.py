import streamlit as st
from pathlib import Path
import pandas as pd
import pickle
import base64 #decodificador imagen

#Configuración basica de streamlit
st.set_page_config(page_title="Estimación Tiempo de Entrega")

### IMAGEN ###
#carga imagen tipo "hero" (cabecera)
#Lee una imagen .jpeg, la convierte a base64 y la pinta con CSS
def hero_image(path:Path, title:str="", subtitle:str=""):
    img_bytes = path.read_bytes()
    b64 = base64.b64encode(img_bytes).decode()
    st.markdown(f"""
                <style>
                    .hero {{
                       position: relative;
                       border-radius: 18px;
                       overflow: hidden;
                       box-shadow: 0 10px 30px rgba(0,0,0,.25);
                       margin-bottom: 18px;
                    }}
                    .hero img {{
                        width: 100%;
                        height: auto;
                        display:block;
                    }}
                    .hero .overlay{{
                        position: absolute;
                        inset: 0;
                        background: linear-gradient(180deg,
                            rgba(0,0,0,.45) 0%,
                            rgba(0,0,0,.15) 45%,
                            rgba(0,0,0,.55) 100%);          
                    }}
                    .hero .text{{
                        position: absolute;
                        left: 22px;
                        bottom: 18px;
                        right: 22px;
                        color:white;
                        text-shadow: 0 2px 12px rgba(0,0,0,.55);
                    }}
                    .hero .title{{
                        font-size: 30px;
                        font-weight: 900;
                        margin: 0;
                        line-height: 1.1;
                    }}
                    .hero .subtitle{{
                        font-size: 14px;
                        margin: 8px 0 0 0;
                        opacity: .95;
                    }}
                /*arreglo formulario sobre fondo*/
                    .card{{
                        background: rgba(255,255,255,0.906);
                        border-radius: 16px;
                        padding:18px;
                        box-shadow: 0 8px 22px rgba(0,0,0,.12);
                        margin-top: 6px;
                    }}
                </style>

                <div class="hero">
                    <img src="data:image/jpeg;base64,{b64}" />
                    <div class="overlay"></div>
                    <div class="text">
                        <div class="title">{title}</div>
                        <div class="subtitle">{subtitle}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
                )
    
### Paths ###
   
"""Rutas del proyecto"""
#APP_DIR: carpeta del archivo app.py (webapp/)
APP_DIR = Path(__file__).resolve().parent

#ROOT_DIR: raíz del repo
ROOT_DIR = APP_DIR.parent

#MODEL_PATH: ruta del .pkl del modelo entrenado
MODEL_PATH = ROOT_DIR / "models" / "random_forest_delivery_time.pkl"

@st.cache_data #decorador cargar en caché para agilizar reruns
def load_df_model(root_dir:Path) -> pd.DataFrame | None: # recibe raíz del repo, devuelve un datafram o None si no puede cargarlo
    path = root_dir / "data" / "processed" / "df_model.csv"#ruta al archivo
    if not path.exists():
        return None# permite usar "text_input" si no existe el archivo
    return pd.read_csv(path)

ref_df = load_df_model(ROOT_DIR)

#eliminamos nulos, espacios y controlar str 
if ref_df is not None:
    city_options = sorted(ref_df["customer_city"].dropna().astype(str).str.strip().unique())
    cat_options = sorted(ref_df["main_product_category"].dropna().astype(str).str.strip().unique())
else:
    city_options, cat_options =None, None

#Carga del modelo con .pkl
#el .pkl contiene un pipeline (preprocesado + modelo)
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

#ruta de imagen
ASSETS_DIR =  APP_DIR / "assets"
HERO_PATH = ASSETS_DIR / "hero.jpeg"

hero_image(HERO_PATH,
           title="Estimación del tiempo de entrega",
           subtitle="Producto mínimo viable para la predicción de entregas a clientes")  

#Títulos y layout              
st.title("Tiempo estimado de entrega")
st.caption("Formulario de entradas")

#"card" visual
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Datos del pedido")

#Form: evita reruns para cada input
with st.form("eta_form"):

#3 columnas para inputs
#las claves que se usan para predecir deben coincidir con el entrenamiento.
    col1, col2, col3 = st.columns(3)

#inputs usuario
    with col1:
        order_status = st.selectbox("Estado del pedido", ["delivered", 
                                                          "shipped", 
                                                          "canceled", 
                                                          "invoiced",
                                                          "processing",
                                                          "approved",
                                                          "created"],
                                    help="situación actual delpedido en el sistema")
        #Lista fija de estados BR sin depender del dataset
        BR_STATES = ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT","PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"]
        customer_state = st.selectbox("Estado", BR_STATES, index=BR_STATES.index("SP"), help="Estado brasileño: SP Sao Paulo, BA Bahía, etc")
       
        #si esxiste city_options selectbox, saca valores unicos de customer_city
        #si no existe, text_input
        if city_options:
            customer_city = st.selectbox("Ciudad",
                                         city_options,
                                         index=city_options.index("sao paulo") if "sao paulo" in city_options else 0,
                                         help="Lista cargada desde csv para evitar errores de escritura")
        else:
            customer_city = st.text_input("Ciudad", value="sao paulo", help="Ciudad del cliente. Ej: sao paulo")

        customer_zip_code_prefix = st.number_input("Código postal (prefijo)", min_value =0, value=10000, step=1, help="Prefijo del codigo postal. Ej: 01000")

    with col2:
        #si existe cat_option selectbox, saca valores unicos de "main_product_category"
        #si no existe, text_input
        if cat_options:
            main_product_category = st.selectbox("Categoría", 
                                                 cat_options, 
                                                 index=cat_options.index("bed_bath_table") if "bed_bath_table" in cat_options else 0,
                                                 help="Lista cargada desde csv para evitar tipos")
        else:
            main_product_category = st.text_input("Categoría", value="bed_bath_table", help="Categoría principal del producto. Ej: bed_bath_table")
        
        total_items = st.number_input("Número de artículos", min_value=1, value=2, step=1, help="Cantidad total de artículos en el pedido")
        total_price = st.number_input("Precio total(R$)", min_value=0.0, value=120.0, step=10.0, help="Suma del precio de los artículos ( sin envío)")
        total_freight = st.number_input("Envio (R$)", min_value=0.0, value=25.0, step=5.0, help="Coste total del envío. Real brasileño")

    with col3:
        payment_value = st.number_input("Pago total", min_value=0.0, value=145.0, step=10.0, help="importe pagado por el cliente (incluyendo envío si aplica)")
        payment_installments = st.number_input("Nº de cuotas", min_value=1, value=1, step=1, help="Número de cuotas del pago")
        geo_lat = st.number_input("Latitud", value=-23.55, step=0.01, format="%.6f",help="Coordenada aproximada del cliente (latitud).")
        geo_lng = st.number_input("Longitud", value=-46.63, step=0.01, format="%.6f",help="Coordenada aproximada del cliente (longitud).")
        #Features extra que el pipeline exige
        purchase_hour = st.number_input("Hora de compra (0–23)", min_value=0, max_value=23, value=12, step=1, help="Hora del día en la que se realizó la compra (0–23).")
        purchase_weekday = st.number_input("Día de la semana", min_value=0, max_value=6, value=2, step=1, help="0=Lun … 6=Dom")
        approval_delay_hours = st.number_input("Retraso de aprobación (horas)", min_value=0.0, value=0.0, step=1.0, help="Horas desde la compra hasta la aprobación del pedido.")

#
#Diccionario con nombres de columnas de entrenamiento
    features = {
        "order_status": order_status,
        "customer_city": customer_city,
        "customer_state": customer_state,
        "customer_zip_code_prefix": customer_zip_code_prefix,
        "main_product_category": main_product_category,
        "total_items": total_items,
        "total_price": total_price,
        "total_freight": total_freight,
        "payment_value": payment_value,
        "payment_installments": payment_installments,
        "geo_lat": geo_lat,
        "geo_lng": geo_lng,
        "purchase_hour": purchase_hour,
        "purchase_weekday": purchase_weekday,
        "approval_delay_hours": approval_delay_hours}

    #submit dentro del form
    submitted = st.form_submit_button("Predecir ETA")

#ver input generado
st.divider()

#ocultar input generado
with st.expander("Ver input"):
    st.json(features)

#predicción
st.divider()
st.subheader("Predicción")

if submitted:
    try:
        model = load_model(MODEL_PATH)
        X = pd.DataFrame([features])# Construimos un DataFrame con 1 fila (un pedido)
        eta_days = float(model.predict(X)[0])
        st.success(f"Tiempo estimado de entrega: **{eta_days:.2f} días**")        

    except Exception as e:
        #si hay errores de columnas
        st.error("Error al predecir. Revisa que las columnas coincidan con el entrenamiento.")
        st.exception(e)


st.markdown("</div>", unsafe_allow_html=True)