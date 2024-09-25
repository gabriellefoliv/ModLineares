import csv
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def carregar_dados(nome_arquivo):
    dias = []
    valores = []
    
    with open(nome_arquivo, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        data_anterior = None
        dia_atual = 0
        
        for linha in reader:
            try:
                data_str = linha[0].strip()
                data_atual = datetime.strptime(data_str, "%Y-%m-%d %H:%M:%S")
                
                if data_anterior is None:
                    data_anterior = data_atual
                    dia_atual = 1
                
                diferenca_dias = (data_atual - data_anterior).days
                dia_atual += diferenca_dias
                dias.append(dia_atual)
                
                valor = float(linha[1].replace(',', '.'))
                valores.append(valor)
                
                data_anterior = data_atual

            except ValueError as e:
                print(f"Erro ao processar linha: {linha} - {e}")
                continue

    return np.array(dias), np.array(valores)

def plotar_graficos(x, y):
    plt.figure(figsize=(10, 5))
    plt.hist(x, bins=20, color='blue', alpha=0.7)
    plt.title('Distribuição da Variável X (Dias)')
    plt.xlabel('Dia')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.hist(y, bins=20, color='green', alpha=0.7)
    plt.title('Distribuição da Variável Y (Valores)')
    plt.xlabel('Valor')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.show()

nome_arquivo = 'petr4.csv'
x, y = carregar_dados(nome_arquivo)
plotar_graficos(x, y)

def calcular_estatisticas(y):
    media = np.mean(y)
    mediana = np.median(y)
    desvio_padrao = np.std(y)
    return media, mediana, desvio_padrao

media_y, mediana_y, desvio_padrao_y = calcular_estatisticas(y)
print(f"Média de Y: {media_y:.2f}")
print(f"Mediana de Y: {mediana_y:.2f}")
print(f"Desvio Padrão de Y: {desvio_padrao_y:.2f}")

def plotar_grafico_xy(x, y):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color='blue', alpha=0.5)
    plt.title('Gráfico XY do Conjunto de Dados')
    plt.xlabel('Dias (X)')
    plt.ylabel('Valores (Y)')
    plt.grid(True)
    plt.show()

plotar_grafico_xy(x, y)

def calcular_coeficiente_correlacao(x, y):
    correlacao = np.corrcoef(x, y)[0, 1]
    return correlacao

coeficiente_correlacao = calcular_coeficiente_correlacao(x, y)
print(f"Coeficiente de Correlação de Pearson: {coeficiente_correlacao:.4f}")

def calcular_reta_quadrados_minimos(x, y):
    n = len(x)
    soma_x = np.sum(x)
    soma_y = np.sum(y)
    soma_xy = np.sum(np.multiply(x, y))
    soma_x2 = np.sum(np.square(x))
    
    beta1 = (n * soma_xy - soma_x * soma_y) / (n * soma_x2 - soma_x**2)
    beta0 = (soma_y - beta1 * soma_x) / n
    
    y_predito = beta0 + beta1 * x
    
    sigma2 = np.sum((y - y_predito) ** 2) / (n - 2)
    
    return beta0, beta1, sigma2, y_predito

beta0, beta1, sigma2, y_predito = calcular_reta_quadrados_minimos(x, y)
print(f"Coeficiente beta0 (intercepto): {beta0:.4f}")
print(f"Coeficiente beta1 (inclinação): {beta1:.4f}")
print(f"Variância do erro (σ²): {sigma2:.4f}")

def plotar_grafico_com_reta(x, y, beta0, beta1):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color='blue', alpha=0.5, label='Dados')
    plt.plot(x, beta0 + beta1 * x, color='red', label='Reta de Quadrados Mínimos', linewidth=2)
    plt.title('Gráfico XY com Reta de Quadrados Mínimos')
    plt.xlabel('Dias (X)')
    plt.ylabel('Valores (Y)')
    plt.legend()
    plt.grid(True)
    plt.show()

plotar_grafico_com_reta(x, y, beta0, beta1)

def calcular_residuos(y, beta0, beta1, x):
    y_predito = beta0 + beta1 * x
    residuos = y - y_predito
    return residuos

residuos = calcular_residuos(y, beta0, beta1, x)

print("Resíduos dos primeiros 10 dados:")
for i in range(10):
    print(f"Resíduo {i + 1}: {residuos[i]:.4f}")

def calcular_anova(y, y_predito):
    n = len(y)
    media_y = np.mean(y)

    ss_total = np.sum((y - media_y) ** 2)
    ss_regressao = np.sum((y_predito - media_y) ** 2)
    ss_residuo = np.sum((y - y_predito) ** 2)

    df_total = n - 1
    df_regressao = 1
    df_residuo = n - 2

    ms_regressao = ss_regressao / df_regressao
    ms_residuo = ss_residuo / df_residuo

    F = ms_regressao / ms_residuo
    alpha = 0.05
    f_critical = stats.f.ppf(1 - alpha, df_regressao, df_residuo)

    return ss_total, ss_regressao, ss_residuo, df_total, df_regressao, df_residuo, ms_regressao, ms_residuo, F, f_critical

ss_total, ss_regressao, ss_residuo, df_total, df_regressao, df_residuo, ms_regressao, ms_residuo, F, f_critical = calcular_anova(y, y_predito)

print("Tabela ANOVA:")
print(f"Soma dos Quadrados Total (SS_total): {ss_total:.4f}")
print(f"Soma dos Quadrados da Regressão (SS_regressao): {ss_regressao:.4f}")
print(f"Soma dos Quadrados dos Resíduos (SS_residuo): {ss_residuo:.4f}")
print(f"Grau de Liberdade Total (df_total): {df_total}")
print(f"Grau de Liberdade da Regressão (df_regressao): {df_regressao}")
print(f"Grau de Liberdade dos Resíduos (df_residuo): {df_residuo}")
print(f"Média dos Quadrados da Regressão (MS_regressao): {ms_regressao:.4f}")
print(f"Média dos Quadrados dos Resíduos (MS_residuo): {ms_residuo:.4f}")
print(f"Estatística F: {F:.4f}")
print(f"Valor Crítico F (0.05): {f_critical:.4f}")

if F > f_critical:
    print("A hipótese nula é rejeitada: o modelo linear é significativo.")
else:
    print("A hipótese nula não é rejeitada: o modelo linear não é significativo.")

def identificar_pontos_influentes(residuos, threshold=2):
    return np.abs(residuos) > threshold * np.std(residuos)

pontos_influentes = identificar_pontos_influentes(residuos)

x_sem_influentes = x[~pontos_influentes]
y_sem_influentes = y[~pontos_influentes]

beta1_sem_influentes = np.sum((x_sem_influentes - np.mean(x_sem_influentes)) * (y_sem_influentes - np.mean(y_sem_influentes))) / np.sum((x_sem_influentes - np.mean(x_sem_influentes)) ** 2)
beta0_sem_influentes = np.mean(y_sem_influentes) - beta1_sem_influentes * np.mean(x_sem_influentes)
y_predito_sem_influentes = beta0_sem_influentes + beta1_sem_influentes * x_sem_influentes

sigma_squared_sem_influentes = np.sum((y_sem_influentes - y_predito_sem_influentes) ** 2) / (len(y_sem_influentes) - 2)

# Estimativas do modelo sem pontos influentes
print("\nEstimativas com dados sem pontos influentes:")
print(f"Coeficiente beta0 (intercepto) sem influentes: {beta0_sem_influentes:.4f}")
print(f"Coeficiente beta1 (inclinação) sem influentes: {beta1_sem_influentes:.4f}")
print(f"Variância do erro (σ²) sem influentes: {sigma_squared_sem_influentes:.4f}")

def plotar_grafico_sem_influentes(x, y, x_sem_influentes, y_sem_influentes, beta0, beta1, beta0_sem_influentes, beta1_sem_influentes):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color='blue', alpha=0.5, label='Dados Originais')
    plt.scatter(x_sem_influentes, y_sem_influentes, color='orange', alpha=0.5, label='Dados Sem Influência')
    plt.plot(x, beta0 + beta1 * x, color='red', label='Reta de Quadrados Mínimos (Original)', linewidth=2)
    plt.plot(x_sem_influentes, beta0_sem_influentes + beta1_sem_influentes * x_sem_influentes, color='green', label='Reta Sem Influência', linewidth=2)
    plt.title('Gráficos com e Sem Pontos Influentes')
    plt.xlabel('Dias (X)')
    plt.ylabel('Valores (Y)')
    plt.legend()
    plt.grid(True)
    plt.show()

plotar_grafico_sem_influentes(x, y, x_sem_influentes, y_sem_influentes, beta0, beta1, beta0_sem_influentes, beta1_sem_influentes)
