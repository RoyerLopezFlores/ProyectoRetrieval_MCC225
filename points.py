import streamlit as st
import torch
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from transformers import AutoTokenizer

# Configuración de la página
st.set_page_config(layout="wide", page_title="Buscador Multimodal 3D")
st.title("🌐 Visualizador de Embeddings 3D e Interfaz de Resultados")

# =====================================================================
# 1. INTEGRACIÓN DE TUS MODELOS Y ENTORNO (Configura esto con tus datos)
# =====================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_models_and_dataset():
    path_tokenizer = "./data/tokenizer_final_retrieval"
    tokenizer = AutoTokenizer.from_pretrained(path_tokenizer)
    
    # 💡 AQUÍ DEBES CARGAR TUS OBJETOS REALES:
    # text_model = ...
    # index_img = ...
    # index_txt = ...
    # test_ds = ...
    
    # Por ahora los dejamos como None para que el script no falle al iniciar,
    # pero reemplázalos con tus cargas reales.
    return tokenizer, None, None, None, None

tokenizer, text_model, index_img, index_txt, test_ds = load_models_and_dataset()


# =====================================================================
# 2. FUNCIÓN DE BÚSQUEDA REAL (Adaptada a tu código original)
# =====================================================================
def perform_retrieval_backend(query_text, top_k=5):
    # Si los modelos no están cargados, simulamos índices para que la app corra de prueba
    if text_model is None:
        return np.array([[0, 1, 2, 3, 4]]), np.array([[10, 11, 12, 13, 14]])
        
    # Tu lógica original:
    tokens = tokenizer([query_text], padding=True, truncation=True, max_length=40, return_tensors="pt")
    ids, masks = tokens["input_ids"].to(DEVICE), tokens["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        query_feat = text_model(ids, masks).cpu().numpy().astype('float32')
    
    D_img, I_img = index_img.search(query_feat, top_k)
    D_txt, I_txt = index_txt.search(query_feat, top_k)
    
    return I_img, I_txt, query_feat


# =====================================================================
# 3. PREPARAR EL ESPACIO 3D (Se ejecuta una sola vez)
# =====================================================================
@st.cache_data
def compute_3d_projections():
    # 💡 Para producción, aquí deberías pasar los vectores reales de TODO tu index o dataset.
    # Simulamos 200 puntos fijos en el espacio para el fondo del gráfico.
    n_samples = 200
    mock_features = np.random.randn(n_samples * 2, 512).astype('float32') 
    
    pca = PCA(n_components=3)
    coords_3d = pca.fit_transform(mock_features)
    
    df = pd.DataFrame(coords_3d, columns=['x', 'y', 'z'])
    df['Tipo'] = ['Imagen en Dataset'] * n_samples + ['Texto en Dataset'] * n_samples
    df['Etiqueta'] = [f"Imagen ID: {i}" for i in range(n_samples)] + [f"Texto ID: {i}" for i in range(n_samples)]
    df['Tamaño'] = 5
    df['Real_Idx'] = list(range(n_samples)) + list(range(n_samples)) # Guardamos el índice real del dataset
    
    return pca, df

pca, df_embeddings = compute_3d_projections()


# =====================================================================
# 4. INTERFAZ DE USUARIO (Sidebar)
# =====================================================================
st.sidebar.header("🔍 Realizar Consulta")
query_text = st.sidebar.text_input("Escribe tu búsqueda:", "A dog running through the grass")
top_k = st.sidebar.slider("Top K resultados", min_value=1, max_value=10, value=5)

if query_text:
    # Ejecutamos tu búsqueda para obtener los índices más cercanos
    best_img_idxs, best_txt_idxs = perform_retrieval_backend(query_text, top_k=top_k)
    
    # Extraemos las listas de índices del formato FAISS (usualmente viene como [[idx1, idx2, ...]])
    img_idxs = best_img_idxs[0]
    txt_idxs = best_txt_idxs[0]
    
    # --- Actualizar gráfico 3D con la Query y los resultados ---
    # (Simulamos la proyección de la query si no hay modelo real cargado)
    if text_model is None:
        query_feat = np.random.randn(1, 512).astype('float32')
        
    query_3d = pca.transform(query_feat)
    
    df_query = pd.DataFrame(query_3d, columns=['x', 'y', 'z'])
    df_query['Tipo'] = 'TU ENTRADA (QUERY)'
    df_query['Etiqueta'] = f"Query: '{query_text}'"
    df_query['Tamaño'] = 18
    df_query['Real_Idx'] = -1
    
    # Duplicamos el dataframe base para modificar los puntos que resultaron elegidos
    df_plot = df_embeddings.copy()
    
    # Marcamos en el dataframe cuáles fueron los índices ganadores para pintarlos en el gráfico 3D
    df_plot.loc[df_plot['Tipo'] == 'Imagen en Dataset', 'Tipo'] = df_plot.loc[df_plot['Tipo'] == 'Imagen en Dataset'].apply(
        lambda row: 'Imagen (¡Resultado TOP K!)' if row['Real_Idx'] in img_idxs else 'Imagen en Dataset', axis=1
    )
    df_plot.loc[df_plot['Tipo'] == 'Texto en Dataset', 'Tipo'] = df_plot.loc[df_plot['Tipo'] == 'Texto en Dataset'].apply(
        lambda row: 'Texto (¡Resultado TOP K!)' if row['Real_Idx'] in txt_idxs else 'Texto en Dataset', axis=1
    )
    
    # Aumentamos el tamaño de los puntos seleccionados
    df_plot.loc[df_plot['Tipo'].str.contains('TOP K'), 'Tamaño'] = 12
    
    # Concatenamos la query
    df_plot = pd.concat([df_plot, df_query], ignore_index=True)
    
    # Dibujar el gráfico interactivo
    st.subheader("🤖 Mapa de Similitud Semántica en 3D")
    fig = px.scatter_3d(
        df_plot, x='x', y='y', z='z',
        color='Tipo',
        hover_name='Etiqueta',
        size='Tamaño',
        size_max=18,
        color_discrete_map={
            'Imagen en Dataset': '#E0E0E0',
            'Texto en Dataset': '#EEEEEE',
            'Imagen (¡Resultado TOP K!)': '#FF5722', # Naranja llamativo
            'Texto (¡Resultado TOP K!)': '#FFC107',  # Amarillo/Ámbar
            'TU ENTRADA (QUERY)': '#00E676'          # Verde brillante
        },
        opacity=0.85
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(aspectmode='cube'))
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("---")
    
    # =====================================================================
    # 5. MOSTRAR LA DATA REAL MÁS CERCANA (Imágenes y Textos)
    # =====================================================================
    st.subheader(f"📊 Datos de los elementos más cercanos (Top {top_k})")
    
    # Creamos dos pestañas o dos columnas grandes. Usaremos columnas para verlo en paralelo.
    col_img, col_txt = st.columns(2)
    
    with col_img:
        st.markdown("### 🖼️ Imágenes más cercanas (Puntos Naranjas)")
        
        if test_ds is not None:
            # Iteramos sobre tus índices reales de imágenes recibidos de FAISS
            for rank, idx in enumerate(img_idxs):
                with st.container():
                    # Obtenemos la imagen de tu estructura de datos original
                    img_data = test_ds.data['image'][idx]
                    
                    # Intentamos buscar si esa imagen tiene un caption asociado en el dataset para dar contexto
                    caption_asociado = test_ds.data['caption_0'][idx] if 'caption_0' in test_ds.data else "Sin descripción de fondo"
                    
                    st.image(img_data, caption=f"Top {rank+1} - [Index {idx}]", use_container_width=True)
                    st.caption(f"**Texto original asociado en dataset:** {caption_asociado}")
                    st.write("") # Espaciador
        else:
            # Vista de diseño si test_ds no está listo
            st.info("Estructura lista. Cuando conectes tu `test_ds`, aquí se renderizarán de forma iterativa las imágenes reales utilizando `st.image()`.")
            st.json({"Índices detectados (FAISS Image)": img_idxs.tolist()})

    with col_txt:
        st.markdown("### 📝 Textos más similares (Puntos Amarillos)")
        
        if test_ds is not None:
            # Iteramos sobre tus índices reales de texto recibidos de FAISS
            for rank, idx in enumerate(txt_idxs):
                # Usamos una caja decorativa (st.info o st.success) para mostrar el texto de forma limpia
                caption = test_ds.data['caption_0'][idx]
                
                st.markdown(f"**Top {rank+1}** (Distancia / Index `{idx}`):")
                st.info(f"💬 {caption}")
        else:
            # Vista de diseño si test_ds no está listo
            st.info("Estructura lista. Cuando conectes tu `test_ds`, aquí aparecerán las cadenas de texto del dataset indexadas por FAISS.")
            st.json({"Índices detectados (FAISS Text)": txt_idxs.tolist()})