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

Através do comando:
```bash
pip install gymnasium[box2d] numpy pandas gymnasium==0.28.1
```
