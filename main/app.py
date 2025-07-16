# BIBLIOTECA NECESSÁRIAS PARA A EXECUÇÃO DO CÓDIGO
# pip install gymnasium[box2d] numpy pandas gymnasium==0.28.1

import gymnasium as gym
import numpy as np
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
n_treinos = 200        # episódios de treinamento

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
    return tuple(np.digitize(s, bins[i]) for i, s in enumerate(estado))

# RETORNA O VALOR Q
def retorna_q(estado, acao):
    return tabela_q.get((estado, acao), 0.0)

# ATUALIZA O VALOR Q(s,a) com base na equação do Q-Learning
def atualiza_q(estado, acao, recompensa, prox_estado):
    q_atual = retorna_q(estado, acao)
    proximo_q_maximo = max([retorna_q(prox_estado, a) for a in range(env.action_space.n)])
    tabela_q[(estado, acao)] = q_atual + alpha * (recompensa + gamma * proximo_q_maximo - q_atual)

########## FASE DE TREINO ##########
print("Iniciando treinamento...\n")

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
        finalizado = pousado or encalhado
        prox_estado = discretizar_estado(prox_estado)

        atualiza_q(estado, acao, recompensa, prox_estado)

        estado = prox_estado
        recompensa_total += recompensa

    # ARMAZENA A RECOMPENSA TOTAL NA JANELA MÓVEL
    janela_recompensas.append(recompensa_total)

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

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        epsilon = max(epsilon_min, epsilon)

    if (treino + 1) % 50 == 0:
        print(
            f"\nEp {treino + 1}/{n_treinos} | Recompensa média última janela: {np.mean(janela_recompensas):.2f} | ε: {epsilon:.4f}")

        # VISUALIZAÇÃO DE UMA AMOSTRA DA TABELA Q
        print(f"\n--- Tabela Q após {treino + 1} episódios ---")
        amostra = list(tabela_q.items())[:10]  # pega até 10 pares (estado, ação)
        for ((estado_print, acao_print), valor_q) in amostra:
            nome_acao = acoes_nome.get(acao_print, "Desconhecida")
            estado_limpo = tuple(int(x) for x in estado_print)
            print(f"Estado: {estado_limpo}, Ação: {acao_print} ({nome_acao}) - Q-Valor: {valor_q:.3f}")
        print("--------------------------------------------\n")

print("\nTreinamento concluído!")
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
        finalizado = pousado or encalhado
        prox_estado = discretizar_estado(prox_estado)

        estado = prox_estado
        recompensa_total += recompensa

        time.sleep(0.02)  # tempo de espera pra ver o pouso

    print(f"Recompensa do Teste {test + 1}: {recompensa_total:.2f}")

env.close()
