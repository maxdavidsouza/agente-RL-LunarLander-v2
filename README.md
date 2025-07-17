# Agente baseado em Aprendizado por Reforço - Teste de LunarLander (v2) [Gymnasium OpenAI]

Este projeto implementa um agente de aprendizado por reforço baseado no algoritmo **Q-Learning** para o ambiente **LunarLander-v2**, utilizando a biblioteca **Gymnasium**.

O objetivo é treinar o agente para pousar com segurança uma nave em uma base lunar simulada, maximizando a recompensa acumulada.

## Sobre o Projeto

- O algoritmo utiliza **discretização do espaço contínuo de estados** para permitir o uso de uma **tabela Q**.
- A política de exploração segue a abordagem de estratégia gulosa, a qual existe um valor **ε**, com decaimento exponencial.
- Após o treinamento, o agente é testado em episódios com renderização em tempo real.

---

## Dependências

Para rodar este projeto, instale as seguintes dependências:

- Python 3.9 ou superior: https://www.python.org/downloads/
- Swig, através do seguinte link: https://www.swig.org/download.html (obs: existem dois downloads, um para Windows e outro para Linux)

E instale as seguintes bibliotecas:
- Pandas
- Numpy
- Gymnasium[box2d] (v0.28.1 para uso do LunarLander-v2)
- Pickle (para salvar a Tabela Q)
- Moviepy (para salvar as gravações do ambiente LunarLander)

Através do comando:
```bash
pip install pygame>=2.5.2 box2d-py==2.3.5 gymnasium==0.28.1 numpy pandas moviepy pickle
```

## Visão Geral do Algoritmo

O agente utiliza **Q-Learning com discretização do espaço de estados**. A Tabela Q armazena os pares `(estado, ação)` e seus respectivos valores esperados de recompensa. Como o ambiente possui estados contínuos, utilizamos uma discretização por bins para torná-los tratáveis.

---

## Hiperparâmetros Utilizados

| Parâmetro       | Descrição                             | Valor        |
|----------------|----------------------------------------|--------------|
| `alpha`        | Taxa de aprendizado                    | `0.1`        |
| `gamma`        | Fator de desconto                      | `0.99`       |
| `epsilon`      | Taxa inicial de exploração             | `1.0`        |
| `epsilon_min`  | Exploração mínima                      | `0.01`       |
| `n_treinos`    | Número de episódios de treinamento     | `2000`       |
| `n_bins`       | Faixas para discretização do estado    | `8`          |

O valor de `epsilon` decai de forma **logística adaptativa**, com base na média de recompensas dos últimos episódios.

---

## Etapas do Código

### 1. **Inicialização**
- Criação do ambiente `LunarLander-v2`
- Definição dos hiperparâmetros
- Discretização das observações contínuas

### 2. **Treinamento**
- Episódios em loop
- Política ε-greedy para escolha de ações
- Atualização da Tabela Q
- Recompensas ajustadas para enfatizar pousos seguros
- Decaimento adaptativo de `epsilon`
- Parada antecipada caso o agente atinja desempenho consistente (dentro da janela de recompensas)

### 3. **Teste**
- Execução de 5 episódios com renderização real (`render_mode="human"`)
- Ações escolhidas pela política greedy (sem exploração)
- Recompensa total exibida ao final de cada episódio
---
### A política ε-greedy é usada durante o treinamento para permitir que o agente experimente ações diferentes e descubra quais são as melhores. Após o treinamento, queremos ver como o agente se comporta quando “confia” no que aprendeu — ou seja, escolhe sempre a ação que acredita ser a melhor (maior valor Q).

---

## Discretização do Espaço de Estados

O estado observado tem 8 dimensões contínuas:

1. Posição horizontal (Entre -1.5 e 1.5)
2. Posição vertical (Entre -0.5 e 1.5)
3. Velocidade horizontal (Entre -2.0 e 2.0)
4. Velocidade vertical (Entre -2.0 e 2.0)
5. Ângulo (Entre -pi e pi)
6. Velocidade angular (Entre -5.0 e 5.0)
7. Contato com o solo esquerdo (0 ou 1)
8. Contato com o solo direito (0 ou 1)

Cada dimensão é dividida em `n_bins = 8` faixas com `np.linspace`, gerando uma tupla de inteiros usada como chave na Tabela Q.

---

## Ações Disponíveis

| Código | Ação                  |
|--------|------------------------|
| `0`    | Ficar parado           |
| `1`    | Motor inferior |
| `2`    | Motor esquerdo |
| `3`    | Motor direito  |

---

## Exemplo de Saída

Durante o treinamento, a Tabela Q é exibida a cada 100 episódios como no exemplo abaixo:

```
--- Tabela Q após X episódios ---
Estado: (3, 4, 3, 4, 4, 4, 1, 1), Ação: 0 (Ficar parado) - Q-Valor: 17.34
Ep X/2000 | Recompensa média última janela: -135.63 | ε: 0.1340
```

Dessa forma, é possível acompanhar o porquê das escolhas feitas pelo agente em cada episódio de treino.

Por exemplo, no caso acima, os dois estados finais "...1, 1)" significam que a nave entrou em contato com o solo,
e a ação do agente foi ficar parado, pois naquele ponto, o valor Q era satisfatório.

---

## Observações

- A recompensa do ambiente era ruidosa, então aplicamos suavizações e bônus/punições para pousos/acidentes.
- A discretização facilita o uso de tabelas mas limita a resolução da política.
- A parada antecipada ocorre quando a recompensa média dos últimos episódios ultrapassa 200.

---

## Demonstração Visual

Após o treinamento, o código executa 5 testes com renderização do ambiente em tempo real. O tempo de espera entre ações é ajustado com `time.sleep(0.02)` para melhor visualização.

---
