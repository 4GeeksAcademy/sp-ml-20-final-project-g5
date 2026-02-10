import streamlit as st
from pathlib import Path
import pandas as pd
import pickle
import base64  # decodificador imagen

# Configuración básica de Streamlit
st.set_page_config(page_title="Estimación Tiempo de Entrega")

# HERO IMAGE (CABECERA)
def hero_image(path: Path, title: str = "", subtitle: str = ""):
    #Lee una imagen .jpeg, la convierte a base64 y la pinta con CSS.
    img_bytes = path.read_bytes()
    b64 = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
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
                display: block;
            }}
            .hero .overlay {{
                position: absolute;
                inset: 0;
                background: linear-gradient(
                    180deg,
                    rgba(0,0,0,.45) 0%,
                    rgba(0,0,0,.15) 45%,
                    rgba(0,0,0,.55) 100%
                );
            }}
            .hero .text {{
                position: absolute;
                left: 22px;
                bottom: 18px;
                right: 22px;
                color: white;
                text-shadow: 0 2px 12px rgba(0,0,0,.55);
            }}
            .hero .title {{
                font-size: 30px;
                font-weight: 900;
                margin: 0;
                line-height: 1.1;
            }}
            .hero .subtitle {{
                font-size: 14px;
                margin: 8px 0 0 0;
                opacity: .95;
            }}

            /* Card para formulario */
            .card {{
                background: rgba(255,255,255,0.906);
                border-radius: 16px;
                padding: 18px;
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
        unsafe_allow_html=True)

# PATHS (rutas)

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODEL_PATH = ROOT_DIR / "models" / "random_forest_delivery_time.pkl"

ASSETS_DIR = APP_DIR / "assets"
HERO_PATH = ASSETS_DIR / "hero.jpeg"


# Cargar el modelo df_model.csv
@st.cache_data
def load_df_model(root_dir: Path) -> pd.DataFrame | None:
    #Carga data/processed/df_model.csv si existe; si no, devuelve None.
    path = root_dir / "data" / "processed" / "df_model.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

#cargar diccionario lookup_zip.csv
@st.cache_data
def load_lookup_zip(root_dir: Path) -> pd.DataFrame | None:
    #Carga data/processed/lookup_zip.csv si existe; si no, devuelve None.
    #columnas esperadas: customer_zip, customer_city, customer_state, geo_lat,geo_lng

    path = root_dir / "data" / "processed" / "lookup_zip.csv"
    if not path.exists():
        return None
    
    df_lu = pd.read_csv(path)

    #normalizar y asegurar tipos en la entrada de datos
    df_lu["customer_zip_code_prefix"] = pd.to_numeric(df_lu["customer_zip_code_prefix"], errors="coerce").astype("Int64")
    df_lu["customer_city"] =  df_lu["customer_city"].astype(str).str.strip().str.lower()
    df_lu["customer_state"] =  df_lu["customer_state"].astype(str).str.strip().str.upper()

    df_lu = df_lu.dropna(subset=["customer_zip_code_prefix", "customer_city", "customer_state", "geo_lat", "geo_lng"]) 
    
    return df_lu.set_index("customer_zip_code_prefix")

ref_df = load_df_model(ROOT_DIR)
lookup_zip = load_lookup_zip(ROOT_DIR)

if ref_df is not None:
    city_options = sorted(ref_df["customer_city"].dropna().astype(str).str.strip().str.lower().unique())
    cat_options = sorted(ref_df["main_product_category"].dropna().astype(str).str.strip().unique())
    default_lat = float(ref_df["geo_lat"].median()) if "geo_lat" in ref_df.columns else -23.50
    default_lng = float(ref_df["geo_lng"].median()) if "geo_lng" in ref_df.columns else -46.60

else:
    city_options, cat_options = None, None
    default_lat, default_lng = -23.50, -46.60

# Cargar MODEL
def load_model(path: Path):
    #Carga el pipeline (prepro + modelo) desde un .pkl.
    with open(path, "rb") as f:
        return pickle.load(f)

# UI
hero_image(HERO_PATH,
           title="Estimación del tiempo de entrega",
           subtitle="Producto mínimo viable para la predicción de entregas a clientes")

st.title("Tiempo estimado de entrega")
st.caption("Formulario de entradas")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Datos del pedido")
st.subheader("Ubicación del cliente")

BR_STATES = ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT","PA","PB","PE", "PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"]

#defaults de seguridad
st.session_state.setdefault("loc_zip", 10000)
st.session_state.setdefault("loc_city", "sao paulo")
st.session_state.setdefault("loc_state", "SP")
st.session_state.setdefault("loc_lat", float(default_lat))
st.session_state.setdefault("loc_lng", float(default_lng))
st.session_state.setdefault("loc_toggle", False)

# Input CP (SIN key -> no bloquea modificaciones en session_state)
new_zip = st.number_input(
    "Código postal (prefijo)",
    min_value=0,
    step=1,
    value=int(st.session_state["loc_zip"]),
    help="Prefijo del código postal. Ej: 01000"
)

# si cambia el CP, actualiza session_state y autocompleta
if int(new_zip) != int(st.session_state["loc_zip"]):
    st.session_state["loc_zip"] = int(new_zip)

    cp_int = int(st.session_state["loc_zip"])
    zip_hit = lookup_zip is not None and cp_int in lookup_zip.index

    if zip_hit:
        row = lookup_zip.loc[cp_int]
        st.session_state["loc_city"] = str(row["customer_city"]).strip()
        st.session_state["loc_state"] = str(row["customer_state"]).strip()
        st.session_state["loc_lat"] = float(row["geo_lat"]) if pd.notna(row["geo_lat"]) else float(default_lat)
        st.session_state["loc_lng"] = float(row["geo_lng"]) if pd.notna(row["geo_lng"]) else float(default_lng)

# mensaje según exista o no
cp_int = int(st.session_state["loc_zip"])
zip_hit = lookup_zip is not None and cp_int in lookup_zip.index

if zip_hit:
    st.success(f"CP encontrado. Autocompletado: {st.session_state['loc_city']} ({st.session_state['loc_state']})")
    st.caption("Lat/Lng autocompletadas. Activa 'Opciones avanzadas' si quieres ver/editar.")
else:
    st.warning("CP no encontrado. Activa 'Opciones avanzadas' para completar/editar ubicación.")

# toggle
st.session_state["loc_toggle"] = st.toggle("Opciones avanzadas (ubicación)", value=st.session_state["loc_toggle"])

# inputs solo si toggle ON
if st.session_state["loc_toggle"]:
    state_idx = BR_STATES.index(st.session_state["loc_state"]) if st.session_state["loc_state"] in BR_STATES else 0
    st.session_state["loc_state"] = st.selectbox("Estado", BR_STATES, index=state_idx)

    if city_options:
        city_idx = city_options.index(st.session_state["loc_city"]) if st.session_state["loc_city"] in city_options else 0
        st.session_state["loc_city"] = st.selectbox("Ciudad", city_options, index=city_idx)
    else:
        st.session_state["loc_city"] = st.text_input("Ciudad", value=st.session_state["loc_city"])

    st.session_state["loc_lat"] = st.number_input("Latitud", value=float(st.session_state["loc_lat"]), step=0.01, format="%.6f")
    st.session_state["loc_lng"] = st.number_input("Longitud", value=float(st.session_state["loc_lng"]), step=0.01, format="%.6f")

# valores finales para el modelo
customer_zip_code_prefix = int(st.session_state["loc_zip"])
customer_city = st.session_state["loc_city"]
customer_state = st.session_state["loc_state"]
geo_lat = float(st.session_state["loc_lat"])
geo_lng = float(st.session_state["loc_lng"])
            
 #FORM solo para inputs del pedido + submit
with st.form("eta_form"):
    col1, col2, col3 = st.columns(3)      
    
    with col1:
        st.markdown("Ubicación del cliente (autocompletada por CP)")
        st.write(f"{customer_city} ({customer_state})")
        st.caption(f"lat={geo_lat:.6f} lng={geo_lng:.6f}")
                

    # columna 2 
    with col2:
        if cat_options:
            main_product_category = st.selectbox("Categoría",
                                                 cat_options,
                                                 index=cat_options.index("bed_bath_table") if "bed_bath_table" in cat_options else 0,
                                                 help="Lista cargada desde df_model.csv para evitar errores de escritura.")
        else:
            main_product_category = st.text_input("Categoría",
                                                  value="bed_bath_table",
                                                  help="Categoría principal del producto. Ej: bed_bath_table")

        total_items = st.number_input("Número de artículos",
                                      min_value=1,
                                      value=2,
                                      step=1,
                                      help="Cantidad total de artículos en el pedido.")
        total_price = st.number_input("Precio total (R$)",
                                      min_value=0.0,
                                      value=120.0,
                                      step=10.0,
                                      help="Suma del precio de los artículos (sin envío).")
        total_freight = st.number_input("Envío (R$)",
                                        min_value=0.0,
                                        value=25.0,
                                        step=5.0,
                                        help="Coste total del envío (Real brasileño).")

    #columna 3 
    with col3:
        payment_value = st.number_input("Pago total (R$)",
                                        min_value=0.0,
                                        value=145.0,
                                        step=10.0,
                                        help="Importe pagado por el cliente (incluyendo envío si aplica).")
        payment_installments = st.number_input("Nº de cuotas",
                                               min_value=1,
                                               value=1,
                                               step=1,
                                               help="Número de cuotas del pago.")
        
        purchase_hour = st.number_input("Hora de compra (0–23)",
                                        min_value=0,
                                        max_value=23,
                                        value=12,
                                        step=1,
                                        help="Hora del día en la que se realizó la compra (0–23)")
        
        purchase_weekday = st.number_input("Día de la semana",
                                           min_value=0,
                                           max_value=6,
                                           value=2,
                                           step=1,
                                           help="0=Lun … 6=Dom")
        approval_delay_hours = st.number_input("Retraso de aprobación (horas)",
                                               min_value=0.0,
                                               value=0.0,
                                               step=1.0,
                                               help="Horas desde la compra hasta la aprobación del pedido.")

    # diccionario con nombres exactos usados en entrenamiento
    features = {"customer_city": customer_city,
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

    submitted = st.form_submit_button("Predecir ETA")

st.divider()


#vista de lo que se manda al modelo
with st.expander("Ver input"):
    st.json(features)

with st.expander("Ver input (DataFrame)"):
    df_input = pd.DataFrame([features])
    st.dataframe(df_input)
    st.write(df_input.dtypes)

st.divider()
st.subheader("Predicción")

if submitted:
    try:
        model = load_model(MODEL_PATH)
        X = pd.DataFrame([features])  # 1 fila = 1 pedido
        eta_days = float(model.predict(X)[0])
        st.success(f"Tiempo estimado de entrega: **{eta_days:.2f} días**")
    except Exception as e:
        st.error("Error al predecir. Revisa que las columnas coincidan con el entrenamiento.")
        st.exception(e)

st.markdown("</div>", unsafe_allow_html=True)
