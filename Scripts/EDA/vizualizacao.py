# DEFININDO FUNÇÕES BASE USADAS DURANTE A EDA

# Paleta de Cores Padrão
cores_churn = {0: "#B0BEC5", 1: "#E53935"} # 0: Retido (Cinza), 1: Churn (Vermelho)


#-------------------------------------------------------------------------------
# FUNÇÃO PARA VARIÁVEIS NUMÉRICAS
def plotar_distribuicao_numerica(df, colunas, target='Exited'):
    """Plota a distribuição (Histograma + KDE) de variáveis numéricas vs Target com foco em Densidade."""
    # Estilo "minimalista"
    sns.set_theme(style="white", palette="muted")

    # Configuração do Grid
    linhas = math.ceil(len(colunas) / 2)  # Cálculo de linhas dinânimco (arredonda para cima)
    fig, axes = plt.subplots(linhas, 2, figsize=(16, 5 * linhas))     # Evita que o Gráfico fique "esmagado"
    axes = axes.flatten()                 # Achata array para sintaxe mais simples

    for i, col in enumerate(colunas):
        sns.histplot(data=df,
                     x=col,
                     hue=target,
                     kde=True,
                     bins=30,
                     palette=cores_churn,
                     common_norm=False, # Normalizar o desbalanceamento para vizualização dos churns = 1
                     stat='density',    # Eixo Y como Densidade (Proporção)
                     alpha=0.8,
                     ax=axes[i])

        # Formatação limpa e padronizada
        sns.despine(ax=axes[i])
        axes[i].set_title(f'Distribuição de {col}', fontsize=14, fontweight='bold', pad=10)
        axes[i].set_xlabel(col, fontsize=12)
        axes[i].set_ylabel('Densidade', fontsize=12)

    # Remove painéis vazios
    for j in range(len(colunas), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------
# 2 - FUNÇÃO PARA VARIÁVEIS CATEGÓRICAS
def plotar_distribuicao_categorica(df, colunas, target='Exited'):
    """Plota a contagem (Countplot) de variáveis categóricas vs Target com visual minimalista."""

    # Estilo minimalista padronizado
    sns.set_theme(style="white", palette="muted")

    # Configuração do Grid
    linhas = math.ceil(len(colunas) / 2) # Cálculo de linhas dinânimco (arredonda para cima)
    fig, axes = plt.subplots(linhas, 2, figsize=(16, 5 * linhas))
    axes = axes.flatten()

    for i, col in enumerate(colunas):
        sns.countplot(data=df,
                      x=col,
                      hue=target,
                      palette=cores_churn,
                      alpha=0.8, # Mantém a mesma textura visual do gráfico numérico
                      ax=axes[i])

        # Formatação limpa e padronizada
        sns.despine(ax=axes[i])
        axes[i].set_title(f'Distribuição de {col}', fontsize=14, fontweight='bold', pad=10)
        axes[i].set_ylabel('Quantidade de Clientes', fontsize=12)
        axes[i].set_xlabel('', fontsize=12) # Mantido vazio para não poluir o eixo X

    # Remove painéis vazios
    for j in range(len(colunas), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


#-------------------------------------------------------------------------------
# FUNÇÃO PARA DETECÇÃO DE OUTLIERS (BOXPLOT)
def plotar_deteccao_outliers(df, colunas, target='Exited'):
    """Plota boxplots horizontais para detecção de outliers em variáveis numéricas vs Target."""

    # Estética consistente com a análise anterior
    sns.set_theme(style="whitegrid") # Whitegrid ajuda a ler as linhas do boxplot
    cores_churn = {0: "#B0BEC5", 1: "#E53935"} # 0: Retido (Cinza), 1: Churn (Vermelho)

    # Configuração do Grid (2 colunas)
    linhas = math.ceil(len(colunas) / 2)
    fig, axes = plt.subplots(linhas, 2, figsize=(16, 4 * linhas))
    axes = axes.flatten()

    for i, col in enumerate(colunas):
        # O boxplot mostra a mediana, quartis e os pontos candidatos a outliers
        sns.boxplot(data=df,
                    x=col,
                    y=target,
                    orient='h',
                    palette=cores_churn,
                    ax=axes[i],
                    hue=target,
                    legend=False)

        # Limpeza visual e formatação
        sns.despine(ax=axes[i])
        axes[i].set_title(f'Detecção de Outliers: {col}', fontsize=13, fontweight='bold', pad=10)
        axes[i].set_xlabel(col, fontsize=12)
        axes[i].set_ylabel('Status de Churn', fontsize=12)

    # Remover eixos extras (painéis vazios)
    for j in range(len(colunas), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------
# FUNÇÃO PARA DIAGNÓSTICO DE CORRELAÇÃO
def plotar_diagnostico_correlacao(df, colunas_num, target='Exited'):
    """Gera um heat map (mapa de calor) para avaliar as correlações do conjunto de entrada"""
    plt.figure(figsize=(15, 7))
    sns.heatmap(corr_matrix,
            annot=True,
            cmap='coolwarm', # Paleta (azul para negativo, vermelho para positivo)
            fmt=".2f",       # Duas casas decimais
            linewidths=0.5,
            cbar_kws={"shrink": .8})
    plt.show()