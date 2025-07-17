import streamlit as st
import numpy as np
import pickle
import os
from io import BytesIO
import matplotlib.pyplot as plt
import urllib.parse
from pathlib import Path
from agente_rl import AgenteRLQLunarLander

st.set_page_config(page_title="Agente - LunarLander-v2", layout="wide")

st.title("Agente LunarLander-v2 com Aprendizado por Reforço")

agente = AgenteRLQLunarLander()

# --- CARREGAR TABELA Q AUTOMATICAMENTE ---
if st.session_state.get("tabela_q", None) is None:
    caminho_padrao = agente.caminho_tabela_q
    if os.path.exists(caminho_padrao):
        with open(caminho_padrao, "rb") as f:
            tabela_q = pickle.load(f)
        st.session_state.tabela_q = tabela_q
        agente.tabela_q = tabela_q
        st.success(f"Tabela Q carregada automaticamente do arquivo '{caminho_padrao}'")
    else:
        st.session_state.tabela_q = agente.tabela_q  # tabela inicial zerada

# CONTROLE DO GRÁFICO DE RECOMPENSAS E LISTA DE EPISÓDIOS GRAVADOS
if "recompensas" not in st.session_state:
    st.session_state.recompensas = None
if "melhores_episodios" not in st.session_state:
    st.session_state.melhores_episodios = agente.melhores_episodios

st.markdown("---")

# INPUT DE TREINO
numero_episodios_treino = st.number_input(
    "Quantidade de episódios para treinar",
    min_value=1,
    max_value=10000,
    value=300,
    step=1
)

if st.button("▶️ Treinar agente"):
    agente.numero_episodios = numero_episodios_treino  # Atualiza o atributo do agente
    recompensas = agente.treinar(exibir_progresso=False)
    st.session_state.recompensas = recompensas
    st.session_state.tabela_q = agente.tabela_q
    st.session_state.melhores_episodios = agente.melhores_episodios
    st.success(f"Treinamento concluído! Foram {numero_episodios_treino} episódios treinados.")

# GRÁFICO DE RECOMPENSAS
if st.session_state.recompensas is not None:
    st.subheader("📊 Gráfico de recompensas por episódio")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(st.session_state.recompensas, label="Recompensa por episódio", alpha=0.5)
    mov_avg = [np.mean(st.session_state.recompensas[max(0, i - 100):i + 1]) for i in range(len(st.session_state.recompensas))]
    ax.plot(mov_avg, label="Média móvel (100 episódios)", color="red")
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Recompensa")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

st.markdown("---")

# VISUALIZADOR DA TABELA Q
if st.session_state.tabela_q is not None:
    st.subheader("🔍 Visualizador da Q-Table")

    tabela_q = st.session_state.tabela_q

    n_bins = (4, 4, 4, 4, 4, 4, 2, 2)
    cols = st.columns(len(n_bins))
    estado = []
    for i, (col, bins) in enumerate(zip(cols, n_bins)):
        with col:
            idx = st.number_input(f"Dim {i+1}", min_value=0, max_value=bins - 1, value=0, key=f"dim_{i}")
            estado.append(idx)
    estado = tuple(estado)

    if estado in np.ndindex(tabela_q.shape[:-1]):
        q_vals = tabela_q[estado]
        melhor_acao = int(np.argmax(q_vals))
        acoes = ["Ficar Parado", "Motor Inferior", "Motor Esquerdo", "Motor Direito"]
        st.table({
            "Ação": acoes,
            "Valor Q": [f"{v:.3f}" for v in q_vals],
            "Melhor?": ["✅" if i == melhor_acao else "" for i in range(4)]
        })
    else:
        st.warning("Estado fora do intervalo da Q-table.")

    st.subheader("📈 Distribuição geral dos valores Q")
    all_qs = tabela_q.reshape(-1, 4)
    flat_qs = all_qs.flatten()
    st.line_chart(flat_qs[:1000])

st.markdown("---")

# BOTÃO PARA GRAVAÇÃO DOS 10 MELHORES EPISÓDIOS
if st.button("🎥 Gravar os 10 melhores episódios (gera vídeos na pasta videos)"):
    agente.melhores_episodios = st.session_state.melhores_episodios
    agente.tabela_q = st.session_state.tabela_q
    agente.gravar_melhores_episodios()
    st.success("Gravação concluída! Vídeos em ./videos/")

st.subheader("🎬 Vídeos dos melhores episódios gravados")

video_dir = agente.diretorio_videos
videos = sorted(Path(video_dir).glob("*.mp4"))

if videos:
    st.write("Clique no link para abrir o vídeo no seu navegador:")
    for video in videos:
        nome_arquivo = video.name
        caminho_relativo = urllib.parse.quote(f"{video_dir}/{nome_arquivo}")
        st.markdown(f'- [{nome_arquivo}](./{caminho_relativo})', unsafe_allow_html=True)
else:
    st.info("Nenhum vídeo gravado encontrado. Grave os melhores episódios primeiro.")

st.markdown("---")

# BOTÃO PARA BAIXAR A TABELA Q ATUAL
buffer = BytesIO()
pickle.dump(st.session_state.tabela_q, buffer)
buffer.seek(0)
st.download_button("💾 Baixar Q-table atual", data=buffer, file_name="tabela_q.pkl")
