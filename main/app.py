# BIBLIOTECA NECESSÁRIAS PARA A EXECUÇÃO DO CÓDIGO
# pip install gymnasium[box2d] numpy pandas gymnasium==0.28.1

import gymnasium as gym
import numpy as np
import random
import time

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
epsilon_decay = 0.995  # redução de exploração a cada episódio
epsilon_min = 0.01     # exploração mínima
n_treinos = 200      # episódios de treinamento

# DISCRETIZANDO O ESPAÇO DE AÇÕES POSSÍVEIS
n_bins = 6  # número de faixas por dimensão
espaco_observavel = env.observation_space
bins = [
    np.linspace(espaco_observavel.low[i], espaco_observavel.high[i], n_bins - 1)
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

    # REDUÇÃO DA TAXA DE EXPLORAÇÃO
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if (treino + 1) % 100 == 0:
        print(f"\n--- Tabela Q após {treino + 1} episódios ---")
        amostra = list(tabela_q.items())[:10]
        for ((estado_print, acao_print), valor_q) in amostra:
            nome_acao = acoes_nome.get(acao_print, "Desconhecida")
            estado_limpo = tuple(int(x) for x in estado_print)
            print(f"Estado: {estado_limpo}, Ação: {acao_print} ({nome_acao}) - Q-Valor: {valor_q:.3f}")
        print("--------------------------------------------\n")
        print(f"Ep {treino + 1}/{n_treinos} | Recompensa: {recompensa_total:.2f} | ε: {epsilon:.3f}")

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
