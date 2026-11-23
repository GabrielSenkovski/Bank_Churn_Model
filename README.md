# Previsão de Churn Bancário

Modelo de machine learning para **prever quais clientes de um banco vão encerrar a conta**, permitindo que a instituição aja antes do cancelamento com campanhas de retenção direcionadas.

**Resultado:** o modelo final (CatBoost) atingiu **ROC-AUC de 0,927** em dados nunca vistos e identificou **83% dos clientes que realmente saíram**.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabrielSenkovski/Bank_Churn_Model/blob/main/Notebooks/ChurnModel.ipynb)

---

## O problema de negócio

Adquirir um cliente novo custa mais caro do que manter um que já existe. Se o banco souber com antecedência quem tem alta probabilidade de sair, pode concentrar ofertas e atendimento nesses clientes em vez de gastar com a base inteira.

O objetivo do projeto é estimar a **probabilidade de churn** de cada cliente e entender **quais fatores explicam a saída**.

## Dados

- **Fonte:** competição do Kaggle [Bank Customer Churn (ICT-U AI)](https://www.kaggle.com/competitions/bank-customer-churn-ict-u-ai/overview)
- **Tamanho:** 15.000 clientes e 14 colunas
- **Alvo:** `Exited` (1 = encerrou a conta)
- **Desbalanceamento:** cerca de 20% dos clientes saíram

Principais variáveis: score de crédito, país, gênero, idade, tempo de relacionamento, saldo, número de produtos, cartão de crédito, se é cliente ativo e salário estimado.

## Principais achados da análise exploratória

- **Excesso de produtos afasta o cliente:** quem tem exatamente 2 produtos é o perfil mais fiel. Com 3 ou 4 produtos, a evasão chega perto de 100%.
- **Idade de risco:** a faixa de 45 a 55 anos concentra a maior taxa de saída. Clientes abaixo de 35 anos são os mais estáveis.
- **O banco perde os clientes mais valiosos:** saldos entre 100 mil e 150 mil têm a maior evasão, enquanto contas com saldo zerado são as mais estáveis.
- **Engajamento importa mais que tempo de casa:** contas inativas saem muito mais. Tempo de relacionamento e salário quase não diferenciam quem sai de quem fica.
- **Diferenças regionais:** clientes da Alemanha têm evasão bem maior que os da França e da Espanha.

**Recomendação:** trocar a estratégia de "vender mais produtos" por "engajar o cliente no uso da conta", com foco no público de meia-idade com saldo alto.

## Abordagem

1. **Divisão treino/teste estratificada (80/20)** antes de qualquer análise, para evitar vazamento de dados.
2. **Engenharia de features** a partir dos achados da EDA, encapsulada em um transformer do scikit-learn:
   - `Active_vs_Products`: cruza atividade da conta com número de produtos
   - `Balance_Salary_Ratio`: quanto da renda anual o cliente mantém no banco
   - `Age_Risk_Zone`: faixas de idade de risco
3. **Pipelines** com pré-processamento separado para modelos clássicos (One-Hot Encoding) e para o CatBoost (categóricas nativas).
4. **Comparação de 5 modelos** com validação cruzada estratificada (5 folds), usando pesos de classe para lidar com o desbalanceamento.
5. **Seleção de variáveis** com apoio de SHAP values.
6. **Otimização de hiperparâmetros** do CatBoost com Optuna (30 tentativas, otimizando ROC-AUC).

## Resultados

### Comparação dos modelos (validação cruzada no treino)

| Modelo | ROC-AUC | Recall | Precisão | F1 |
|---|---|---|---|---|
| **CatBoost** | **0,927** | 0,816 | 0,632 | 0,712 |
| Regressão Logística | 0,924 | 0,844 | 0,598 | 0,700 |
| LightGBM | 0,923 | 0,810 | 0,646 | 0,718 |
| Random Forest | 0,919 | 0,634 | 0,792 | 0,704 |
| XGBoost | 0,911 | 0,756 | 0,668 | 0,709 |

### Modelo final no conjunto de teste (3.000 clientes)

| Métrica | Valor |
|---|---|
| ROC-AUC | 0,927 |
| Recall (churn) | 83% |
| Precisão (churn) | 60% |
| Acurácia | 85% |

**Em termos de negócio:** dos 615 clientes que saíram no teste, o modelo sinalizou 513. Dos 849 clientes que ele apontou como risco, 336 eram falsos alarmes. Priorizei o recall porque, numa campanha de retenção, deixar de abordar quem vai sair costuma custar mais do que fazer contato com quem ia ficar.

![Matriz de confusão](imagens/matriz_confusao.png)

### Variáveis mais importantes

Número de produtos e idade dominam a decisão do modelo, o que confirma os achados da EDA.

![Importância das variáveis](imagens/importancia_variaveis.png)

## Estrutura do repositório

```
Bank_Churn_Model/
├── Datasets/            # train.csv, test.csv e sample_submission.csv do Kaggle
├── Notebooks/
│   ├── Churn_EDA.ipynb   # análise exploratória e diagnóstico de negócio
│   └── ChurnModel.ipynb  # features, comparação de modelos, tuning e avaliação
├── Scripts/
│   └── EDA/visualizacao.py  # funções de gráficos usadas na EDA
├── imagens/             # gráficos usados neste README
└── requirements.txt
```

## Como executar

O jeito mais simples é abrir o notebook do modelo no Google Colab pelo botão no topo da página. Os dados são carregados direto deste repositório.

Para rodar localmente:

```bash
git clone https://github.com/GabrielSenkovski/Bank_Churn_Model.git
cd Bank_Churn_Model
pip install -r requirements.txt
jupyter notebook
```

> Os notebooks foram desenvolvidos no Colab. Para rodar fora dele, remova as células de `drive.mount` e ajuste o caminho `project_path` no notebook de EDA.

## Tecnologias

Python · pandas · NumPy · scikit-learn · imbalanced-learn · CatBoost · XGBoost · LightGBM · Optuna · SHAP · Matplotlib · Seaborn

## Próximos passos

- Ajustar o limiar de decisão de acordo com o custo de uma ação de retenção
- Mover o pré-processamento e o modelo para os scripts em `Scripts/`
- Gerar as previsões para o `test.csv` e submeter na competição

---

Feito por [Gabriel Luiz Senkovski](https://github.com/GabrielSenkovski) · [LinkedIn](https://www.linkedin.com/in/gabriel-luiz-senkovski-729a93242/)
