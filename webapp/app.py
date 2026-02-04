import streamlit as st
from pathlib import Path
import pandas as pd
import pickle
import base64 #decodificador imagen


st.set_page_config(page_title="Estimación Tiempo de Entrega")

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
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODEL_PATH = ROOT_DIR / "models" / "random_forest_delivery_time.pkl"

def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

ASSETS_DIR =  APP_DIR / "assets"
HERO_PATH = ASSETS_DIR / "hero.jpeg"

hero_image(HERO_PATH,
           title="Estimación del tiempo de entrega",
           subtitle="Producto mínimo viable para la predicción de entregas a clientes")  



                
st.title("Tiempo estimado de entrega")
st.caption("Formulario de entradas")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Datos del pedido")

col1, col2, col3 = st.columns(3)

with col1:
    order_status = st.selectbox("order_status", ["delivered", 
                                                 "shipped", 
                                                 "canceled", 
                                                 "invoiced",
                                                 "processing",
                                                 "approved",
                                                 "created"])
    customer_state = st.text_input("customer_state (ej.: SP)", value="SP")
    customer_city = st.text_input("customer city", value = "sao paulo")
    customer_zip_code_prefix = st.number_input("customer_zip_code_prefix", min_value =0, value=10000, step=1)

with col2:
    main_product_category = st.text_input("main_product_category", value="bed_bath_table")
    total_items = st.number_input("total_items", min_value=1, value=2, step=1)
    total_price = st.number_input("total_price", min_value=0.0, value=120.0, step=10.0)
    total_freight = st.number_input("total_freight", min_value=0.0, value=25.0, step=5.0)

with col3:
    payment_value = st.number_input("payment_value", min_value=0.0, value=145.0, step=10.0)
    payment_installments = st.number_input("payment_installments", min_value=1, value=1, step=1)
    geo_lat = st.number_input("geo_lat", value=-23.55, step=0.01, format="%.6f")
    geo_lng = st.number_input("geo_lng", value=-46.63, step=0.01, format="%.6f")
    purchase_hour = st.number_input("purchase_hour (0-23)", min_value=0, max_value=23, value=12, step=1)
    purchase_weekday = st.number_input("purchase_weekday (0=Lun ... 6=Dom)", min_value=0, max_value=6, value=2, step=1)
    approval_delay_hours = st.number_input("approval_delay_hours", min_value=0.0, value=0.0, step=1.0)

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

st.divider()
st.subheader("Input generado")
st.write(features)

st.divider()
st.subheader("Predicción")

if st.button("Predecir ETA"):
    try:
        model = load_model(MODEL_PATH)

        X = pd.DataFrame([features])#1 fila
        
        eta_days = float(model.predict(X)[0])

        st.success(f"Tiempo estimado de entrega: **{eta_days:.2f} días**")

    except Exception as e:
        st.error("Error al predecir. Revisa que las columnas coincidan con el entrenamiento.")
        st.exception(e)


st.markdown("</div>", unsafe_allow_html=True)