# BIBLIOTECA NECESSÁRIAS PARA A EXECUÇÃO DO CÓDIGO
# pip install gymnasium[box2d] numpy pandas gymnasium==0.28.1 matplotlib

import gymnasium as gym
import numpy as  np
import matplotlib.pyplot as plt
import random
import time
from collections import deque  # Para janela móvel

# CRIANDO O AMBIENTE DE TESTE COM O GYMNASIUM
env = gym.make("LunarLander-v2")

# DEFININDO AS AÇÕES POSSÍVEIS DO AGENTE
acoes_nome = {
    0: "Ficar parado",
    1: "Motor inferior",
    2: "Motor esquerdo",
    3: "Motor direito"
}

# DEFININDO OS HIPERPARÂMETROS DO ALGORITMO DE APRENDIZADO
alpha = 0.1            # taxa de aprendizado
gamma = 0.99           # fator de desconto
epsilon = 1.0          # taxa de exploração inicial
epsilon_min = 0.01     # exploração mínima
n_treinos = 2000        # episódios de treinamento

# USANDO JANELAS DE RECOMPENSA PARA DECAIMENTO ADAPTATIVO
janela_recompensas = deque(maxlen=20)
epsilon_decay_min = 0.99    # decaimento mínimo padrão por episódio
epsilon_decay_max = 0.9995  # decaimento mais lento quando agente melhora

# DISCRETIZAÇÃO DO ESPAÇO DE ESTADOS
# Posição horizontal (Entre -1.5 e 1.5)
# Posição vertical (Entre -0.5 e 1.5)
# Velocidade horizontal (Entre -2.0 e 2.0)
# Velocidade vertical (Entre -2.0 e 2.0)
# Ângulo (Entre -pi e pi)
# Velocidade angular (Entre -5.0 e 5.0)
# Contato com o solo esquerdo (0 ou 1)
# Contato com o solo direito (0 ou 1)
n_bins = 8  # número de faixas por dimensão oferecidas no ambiente
espaco_observavel = env.observation_space

# FUNÇÃO PARA CRIAR BINS PERCENTIS
def cria_bins_percentis(low, high, n_bins):
    return np.linspace(low, high, n_bins - 1)

bins = [
    cria_bins_percentis(espaco_observavel.low[i], espaco_observavel.high[i], n_bins)
    for i in range(espaco_observavel.shape[0])
]

# INICIALIZANDO A TABELA-Q
tabela_q = {}

# RETORNA UM ESTADO DISCRETIZADO
def discretizar_estado(estado):
    estado_clipado = np.clip(estado, espaco_observavel.low, espaco_observavel.high)
    return tuple(np.digitize(estado_clipado[i], bins[i]) for i in range(len(bins)))

# RETORNA O VALOR Q
def retorna_q(estado, acao):
    return tabela_q.get((estado, acao), 0.0)

# ATUALIZA O VALOR Q(s,a) com base na equação do Q-Learning
def atualiza_q(estado, acao, recompensa, prox_estado):
    q_atual = retorna_q(estado, acao)
    proximo_q_maximo = max([retorna_q(prox_estado, a) for a in range(env.action_space.n)])
    tabela_q[(estado, acao)] = q_atual + alpha * (recompensa + gamma * proximo_q_maximo - q_atual)

# Lista para armazenar recompensas totais para gráfico
recompensas_totais = []

########## FASE DE TREINO ##########
print("Iniciando treinamento...\n")
tempo_inicial = time.time()

for treino in range(n_treinos):
    estado, _ = env.reset()
    estado = discretizar_estado(estado)
    finalizado = False
    recompensa_total = 0

    while not finalizado:
        # ESCOLHA DE AÇÃO COM BASE NA ESTRATÉGIA GULOSA
        if random.random() < epsilon:
            acao = env.action_space.sample()
        else:
            valores_q = [retorna_q(estado, a) for a in range(env.action_space.n)]
            acao = int(np.argmax(valores_q))

        prox_estado, recompensa, pousado, encalhado, _ = env.step(acao)
        # SUAVIZANDO PUNIÇÕES E VALORIZANDO POUSOS OU ACIDENTES
        recompensa = np.clip(recompensa, -100, 100)
        if pousado:
            recompensa += 100
        elif encalhado:
            recompensa -= 100

        finalizado = pousado or encalhado
        prox_estado = discretizar_estado(prox_estado)

        atualiza_q(estado, acao, recompensa, prox_estado)

        estado = prox_estado
        recompensa_total += recompensa

    # ARMAZENA A RECOMPENSA TOTAL NA JANELA MÓVEL E NA LISTA
    janela_recompensas.append(recompensa_total)
    recompensas_totais.append(recompensa_total)

    # DECAY ADAPTATIVO DA TAXA DE EXPLORAÇÃO
    if len(janela_recompensas) == janela_recompensas.maxlen:
        media_recompensa = np.mean(janela_recompensas)
        # Se a recompensa média ultrapassar um limiar, decaimento é mais lento (exploração mais conservada)
        if media_recompensa > 50:  # Ajuste esse valor conforme o desempenho do ambiente
            epsilon_decay = epsilon_decay_max
        else:
            epsilon_decay = epsilon_decay_min
    else:
        epsilon_decay = epsilon_decay_min

    # ADOTANDO A ESTRATÉGIA DE DECAIMENTO LOGÍSTICO
    epsilon = epsilon_min + (1.0 - epsilon_min) * np.exp(-0.01 * treino)

    if (treino + 1) % 100 == 0:
        media_20 = np.mean(janela_recompensas)
        variancia_20 = np.var(janela_recompensas)
        tempo_decorrido = time.time() - tempo_inicial
        print(f"\n--- Tabela Q após {treino + 1} episódios ---")
        amostra = list(tabela_q.items())[:10]  # pega até 10 pares (estado, ação)
        for ((estado_print, acao_print), valor_q) in amostra:
            nome_acao = acoes_nome.get(acao_print, "Desconhecida")
            estado_limpo = tuple(int(x) for x in estado_print)
            print(f"Estado: {estado_limpo}, Ação: {acao_print} ({nome_acao}) - Q-Valor: {valor_q:.3f}")
        print("--------------------------------------------\n")
        print(f"Ep {treino + 1}/{n_treinos} | Recompensa média últimos 20: {media_20:.2f} | Variância: {variancia_20:.2f} | ε: {epsilon:.4f}")
        print(f"Tempo decorrido até agora: {tempo_decorrido:.2f} segundos\n")

        # INTERROMPENDO TREINOS AO ALCANÇAR UMA RECOMPENSA SATISFATÓRIA
        if media_20 > 200:
            print(f"Agente aprendeu! Encerrando treino no episódio {treino + 1}")
            break

print("\nTreinamento concluído!")

print("\nIniciando teste em tempo real com renderização e gravação de vídeo...")

# Criando ambiente com renderização RGB para gravação
video_folder = "./videos"
env = gym.make("LunarLander-v2", render_mode="rgb_array")

# Adicionando wrapper para gravação de vídeo
env = gym.wrappers.RecordVideo(
    env,
    video_folder=video_folder,
    episode_trigger=lambda ep: True,
    name_prefix="qlearning-lander"
)

n_testes = 5

for test in range(n_testes):
    estado, _ = env.reset()
    estado = discretizar_estado(estado)
    finalizado = False
    recompensa_total = 0
    print(f"\nFase de Teste {test + 1}")

    while not finalizado:
        valores_q = [retorna_q(estado, a) for a in range(env.action_space.n)]
        acao = int(np.argmax(valores_q))

        prox_estado, recompensa, pousado, encalhado, _ = env.step(acao)

        recompensa = np.clip(recompensa, -100, 100)
        if pousado:
            recompensa += 100
        elif encalhado:
            recompensa -= 100

        finalizado = pousado or encalhado
        prox_estado = discretizar_estado(prox_estado)

        estado = prox_estado
        recompensa_total += recompensa

    print(f"Recompensa do Teste {test + 1}: {recompensa_total:.2f}")

env.close()

print(f"\n🎥 Vídeo gravado salvo na pasta: {video_folder}")

# PLOTANDO O GRÁFICO DE DESEMPENHO
plt.plot(recompensas_totais)
plt.xlabel("Episódio")
plt.ylabel("Recompensa Total")
plt.title("Desempenho do Agente durante o Treinamento")
plt.grid(True)
plt.show()

env.close()

########## FASE DE TESTE ##########
print("\nIniciando teste em tempo real com renderização...")

env = gym.make("LunarLander-v2", render_mode="human")
n_testes = 5

for test in range(n_testes):
    estado, _ = env.reset()
    estado = discretizar_estado(estado)
    finalizado = False
    recompensa_total = 0
    print(f"\nFase de Teste {test + 1}")

    while not finalizado:
        valores_q = [retorna_q(estado, a) for a in range(env.action_space.n)]
        acao = int(np.argmax(valores_q))

        prox_estado, recompensa, pousado, encalhado, _ = env.step(acao)
        # SUAVIZANDO PUNIÇÕES E VALORIZANDO POUSOS OU ACIDENTES
        recompensa = np.clip(recompensa, -100, 100)
        if pousado:
            recompensa += 100
        elif encalhado:
            recompensa -= 100
        finalizado = pousado or encalhado
        prox_estado = discretizar_estado(prox_estado)

        estado = prox_estado
        recompensa_total += recompensa

        time.sleep(0.05)  # Pausa para melhor visualização do pouso

    print(f"Recompensa do Teste {test + 1}: {recompensa_total:.2f}")

env.close()
