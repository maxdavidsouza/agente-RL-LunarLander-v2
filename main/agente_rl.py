import gymnasium as gym
import numpy as np
import random
import os
import shutil
import pickle
from collections import deque
import matplotlib.pyplot as plt

class AgenteRLQLunarLander:
    def __init__(
        self,
        caminho_tabela_q="main/tabela_q.pkl",
        diretorio_videos="main/videos",
        numero_episodios=300,
        taxa_aprendizado_inicial=1.0,
        decadencia_aprendizado=0.00016,
        taxa_exploracao_inicial=0.005,
        decadencia_exploracao=1e-6,
        taxa_aprendizado_minima=0.0,
        taxa_exploracao_minima=0.0,
        fator_desconto=1.0
    ):
        # INICIALIZANDO A TABELA Q E A PASTA DE GRAVAÇÕES
        self.caminho_tabela_q = caminho_tabela_q
        self.diretorio_videos = diretorio_videos
        self.numero_episodios = numero_episodios
        self.taxa_aprendizado_inicial = taxa_aprendizado_inicial
        self.decadencia_aprendizado = decadencia_aprendizado
        self.taxa_exploracao_inicial = taxa_exploracao_inicial
        self.decadencia_exploracao = decadencia_exploracao
        self.taxa_aprendizado_minima = taxa_aprendizado_minima
        self.taxa_exploracao_minima = taxa_exploracao_minima
        self.fator_desconto = fator_desconto

        # DISCRETIZAÇÃO DO ESPAÇO DE ESTADOS
        self.numero_bins = (4, 4, 4, 4, 4, 4, 2, 2)
        self.limites_inferiores = [-0.1, -0.2, -0.2, -0.2, -0.2, -0.2, 0.0, 0.0]
        self.limites_superiores = [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0]

        # Inicializa a tabela Q e outras variáveis
        self.tabela_q = None
        self.melhores_episodios = []

        # APAGANDO A PASTA DE VIDEOS, CASO EXISTA, E CRIANDO UMA NOVA PASTA
        if os.path.exists(self.diretorio_videos):
            shutil.rmtree(self.diretorio_videos)
        os.makedirs(self.diretorio_videos, exist_ok=True)

        # CARREGA A TABELA Q SE EXISTIR
        if os.path.exists(self.caminho_tabela_q):
            with open(self.caminho_tabela_q, "rb") as f:
                self.tabela_q = pickle.load(f)
            print("Tabela Q carregada com sucesso. Pulando fase de treinamento.")
        else:
            # INICIALIZA A TABELA Q ZERADA
            self.tabela_q = np.zeros(self.numero_bins + (4,))  # 4 ações no LunarLander

    # FUNÇÃO PARA CRIAR BINS PERCENTIS
    def obter_bin(self, valor, limite_inferior, limite_superior, bins):
        valor = min(max(valor, limite_inferior), limite_superior)
        indice_bin = int((valor - limite_inferior) / (limite_superior - limite_inferior) * bins)
        return min(indice_bin, bins - 1)

    # RETORNA UM ESTADO DISCRETIZADO
    def discretizar_estado(self, estado):
        return tuple(
            self.obter_bin(s, li, ls, b)
            for s, li, ls, b in zip(estado, self.limites_inferiores, self.limites_superiores, self.numero_bins)
        )

    # RETORNA A TAXA DE APRENDIZADO (alpha)
    def taxa_aprendizado(self, episodio):
        return max(self.taxa_aprendizado_minima, self.taxa_aprendizado_inicial * (1 - self.decadencia_aprendizado) ** episodio)

    # RETORNA O MELHOR Q(estado, acao) COM BASE NA ESTRATÉGIA GULOSA (e-greedy)
    def selecionar_acao(self, estado, episodio):
        taxa_exploracao = max(self.taxa_exploracao_minima, self.taxa_exploracao_inicial * (1 - self.decadencia_exploracao) ** episodio)
        if random.random() < taxa_exploracao:
            return random.randint(0, 3)
        else:
            return int(np.argmax(self.tabela_q[estado]))

    # ATUALIZA O VALOR Q(estado, acao) com base na equação do Q-Learning
    def atualizar_tabela_q(self, estado, acao, recompensa, proximo_estado, episodio):
        lr = self.taxa_aprendizado(episodio)
        futuro_q = np.max(self.tabela_q[proximo_estado])
        self.tabela_q[estado][acao] = (1 - lr) * self.tabela_q[estado][acao] + lr * (recompensa + self.fator_desconto * futuro_q)

    # FASE DE TREINO
    def treinar(self, exibir_progresso=True):
        ambiente = gym.make("LunarLander-v2")
        recompensas = []
        media_movel = deque(maxlen=100)
        self.melhores_episodios = []

        print("Iniciando treinamento...\n")

        for episodio in range(self.numero_episodios):
            observacao, _ = ambiente.reset()
            estado = self.discretizar_estado(observacao)
            terminado = False
            recompensa_total = 0

            while not terminado:
                acao = self.selecionar_acao(estado, episodio)
                proxima_obs, recompensa, terminado1, terminado2, _ = ambiente.step(acao)
                terminado = terminado1 or terminado2
                proximo_estado = self.discretizar_estado(proxima_obs)
                self.atualizar_tabela_q(estado, acao, recompensa, proximo_estado, episodio)
                estado = proximo_estado
                recompensa_total += recompensa

            recompensas.append(recompensa_total)
            media_movel.append(recompensa_total)

            # GUARDA OS 10 MELHORES EPISÓDIOS
            if len(self.melhores_episodios) < 10 or recompensa_total > min(self.melhores_episodios, key=lambda x: x[1])[1]:
                self.melhores_episodios.append((episodio, recompensa_total))
                self.melhores_episodios = sorted(self.melhores_episodios, key=lambda x: x[1], reverse=True)[:10]

            # MOSTRA PROGRESSO A CADA 100 EPISÓDIOS
            if exibir_progresso and (episodio + 1) % 100 == 0:
                media = np.mean(media_movel)
                print(f"Episódio {episodio + 1}/{self.numero_episodios} | Média últimos 100: {media:.2f}")

        ambiente.close()

        # SALVA A TABELA Q
        with open(self.caminho_tabela_q, "wb") as f:
            pickle.dump(self.tabela_q, f)

        if exibir_progresso:
            print("\nTabela Q salva com sucesso!")

        return recompensas

    # EXIBE O GRÁFICO DE RECOMPENSAS TOTAIS (EM MÉDIA) E POR EPISÓDIO
    def plotar_recompensas(self, recompensas):
        plt.figure(figsize=(12, 5))
        plt.plot(recompensas, label="Recompensa por episódio", alpha=0.5)
        plt.plot(
            [np.mean(recompensas[max(0, i - 100):i + 1]) for i in range(len(recompensas))],
            label="Média móvel (100 episódios)",
            color="red"
        )
        plt.xlabel("Episódio")
        plt.ylabel("Recompensa")
        plt.title("Recompensa por Episódio (Q-Learning - LunarLander-v2)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("grafico_recompensa.png")
        plt.show()

    # GRAVA OS 10 MELHORES EPISÓDIOS COM A TABELA Q
    def gravar_melhores_episodios(self):
        if not self.melhores_episodios:
            # CASO NÃO EXISTA UMA LISTA COM OS 10 MELHORES EPISÓDIOS, CRIA UMA PADRÃO
            self.melhores_episodios = [(i, 0) for i in range(10)]

        print("\nGravando os 10 melhores episódios...")

        for num_episodio, _ in self.melhores_episodios:
            ambiente = gym.make("LunarLander-v2", render_mode="rgb_array")
            ambiente = gym.wrappers.RecordVideo(
                ambiente,
                video_folder=self.diretorio_videos,
                name_prefix=f"ep{num_episodio}",
                episode_trigger=lambda e: True,
                disable_logger=True
            )

            observacao, _ = ambiente.reset()
            estado = self.discretizar_estado(observacao)
            terminado = False

            while not terminado:
                acao = int(np.argmax(self.tabela_q[estado]))
                proxima_obs, recompensa, terminado1, terminado2, _ = ambiente.step(acao)
                terminado = terminado1 or terminado2
                estado = self.discretizar_estado(proxima_obs)

            ambiente.close()
            print(f"Gravado episódio {num_episodio}")

        print(f"\nTodos os vídeos estão na pasta ./{self.diretorio_videos}/")
