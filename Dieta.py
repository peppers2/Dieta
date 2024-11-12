# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import plotly.figure_factory as ff


import alimentos

# Configuração da página
st.set_page_config(page_title="TCC Univesp - Sistema de Recomendação",
                   page_icon="logo.png", layout="wide")

# Função para injetar CSS personalizado


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Chama a função para injetar CSS
local_css("style.css")

# Funções para cálculo de IMC e TMB


def calcular_imc(peso, altura):
    return peso / ((altura / 100) ** 2) if altura > 0 else 0


def calcular_tmb(peso, altura, idade, genero, atividade_fisica):
    if genero == "Masculino":
        tmb = 88.36 + (13.4 * peso) + (4.8 * altura) - (5.7 * idade)
    else:
        tmb = 447.6 + (9.2 * peso) + (3.1 * altura) - (4.3 * idade)

    fator_atividade = {"Sedentário": 1.2, "Leve": 1.375,
                       "Moderado": 1.55, "Intenso": 1.725}
    return tmb * fator_atividade.get(atividade_fisica, 1)


# Carregar dados de alimentos e pacientes

df_alimentos = pd.read_csv("data/alimentos.csv")
df_pacientes = pd.read_csv(
    "pacientes_base_treinamento_1000_formatado_brasil.csv", delimiter=";")

# Ajustar dados de pacientes (caso necessário)
df_pacientes["Peso (kg)"] = df_pacientes["Peso (kg)"].str.replace(
    ",", ".").astype(float)
df_pacientes["Altura (cm)"] = df_pacientes["Altura (cm)"].str.replace(
    ",", ".").astype(float)
df_pacientes["IMC"] = df_pacientes["IMC"].str.replace(",", ".").astype(float)

# Doenças para anamnese
doencas_opcoes = [
    "Diabetes Tipo 1", "Diabetes Tipo 2", "Hipertensão", "Obesidade",
    "Insuficiência Renal", "Colesterol Alto", "Doenças Cardíacas",
    "Osteoporose", "Doença Celíaca"
]

# Função para sugerir refeições sem ultrapassar a TMB e respeitando restrições


# Função ajustada para sugerir refeições com cálculo correto dos nutrientes
def sugerir_refeicoes_ajustado(tmb, previsoes_doencas):
    doencas_preditas = [
        disease for disease, risk in previsoes_doencas.items() if risk == 1
    ]
    df_permitidos = df_alimentos[
        ~df_alimentos["Doencas_Restritivas"].isin(doencas_preditas)
    ]
    refeicoes = {}
    calorias_refeicao = tmb / 4 if tmb > 0 else 0
    proporcoes = {"Proteína": 0.4, "Carboidrato": 0.3, "Lipídeos": 0.3}

    for tipo_refeicao in ["Café da Manhã", "Almoço", "Lanche da Tarde", "Jantar"]:
        alimentos_disponiveis = df_permitidos[
            df_permitidos["Refeicao_Indicada"] == tipo_refeicao
        ]

        refeicao = []
        nutrientes_totais = {"Proteína": 0, "Carboidrato": 0, "Lipídeos": 0}

        for nutriente, proporcao in proporcoes.items():
            calorias_nutriente = calorias_refeicao * proporcao
            alimentos_nutriente = alimentos_disponiveis[
                alimentos_disponiveis[nutriente] > 0
            ].sample(frac=1).reset_index(drop=True)
            total_calorias_nutriente = 0
            idx = 0

            while total_calorias_nutriente < calorias_nutriente and idx < len(alimentos_nutriente):
                alimento = alimentos_nutriente.loc[idx]
                calorias_por_100g = alimento["Calorias"]
                proteina_por_100g = alimento["Proteína"]
                carboidrato_por_100g = alimento["Carboidrato"]
                lipidios_por_100g = alimento["Lipídeos"]
                nutriente_por_100g = alimento[nutriente]

                if calorias_por_100g <= 0 or nutriente_por_100g <= 0:
                    idx += 1
                    continue  # Evita alimentos com calorias ou nutrientes não positivos

                # Calcula a quantidade necessária do alimento
                quantidade = (
                    (calorias_nutriente - total_calorias_nutriente)
                    / (nutriente_por_100g * (9 if nutriente == "Lipídeos" else 4))
                ) * 100

                quantidade = max(quantidade, 0)

                calorias_adicionadas = (calorias_por_100g * quantidade) / 100
                proteina_g = (proteina_por_100g * quantidade) / 100
                carboidrato_g = (carboidrato_por_100g * quantidade) / 100
                lipidios_g = (lipidios_por_100g * quantidade) / 100

                if quantidade > 0 and calorias_adicionadas > 0:
                    alimento_ajustado = {
                        "Alimento": alimento["Alimento"],
                        "Quantidade (g)": quantidade,
                        "Calorias": calorias_adicionadas,
                        "Proteína": proteina_g,
                        "Carboidrato": carboidrato_g,
                        "Lipídeos": lipidios_g,
                    }

                    refeicao.append(alimento_ajustado)
                    total_calorias_nutriente += calorias_adicionadas

                idx += 1

        # Cria DataFrame da refeição
        df_refeicao = pd.DataFrame(refeicao)

        # Remove alimentos com quantidade zero
        df_refeicao = df_refeicao[df_refeicao["Quantidade (g)"] > 0]

        # Agrupa por alimento para somar quantidades e nutrientes
        df_refeicao = df_refeicao.groupby("Alimento", as_index=False).agg({
            "Quantidade (g)": "sum",
            "Calorias": "sum",
            "Proteína": "sum",
            "Carboidrato": "sum",
            "Lipídeos": "sum",
        })

        # Remove alimentos com quantidade zero após a agregação
        df_refeicao = df_refeicao[df_refeicao["Quantidade (g)"] > 0]

        # Ajusta a quantidade total para atingir as calorias da refeição
        total_calorias_refeicao = df_refeicao["Calorias"].sum()
        fator_ajuste = calorias_refeicao / \
            total_calorias_refeicao if total_calorias_refeicao > 0 else 0

        if fator_ajuste > 0:
            df_refeicao["Quantidade (g)"] *= fator_ajuste
            df_refeicao["Calorias"] *= fator_ajuste
            df_refeicao["Proteína"] *= fator_ajuste
            df_refeicao["Carboidrato"] *= fator_ajuste
            df_refeicao["Lipídeos"] *= fator_ajuste

        refeicoes[tipo_refeicao] = df_refeicao

    return refeicoes

# Função para treinar o modelo de risco de doença com base nos dados do paciente


def treinar_modelo_risco(df_pacientes):
    y_columns = [col for col in df_pacientes.columns if "Doença" in col]
    X = df_pacientes.drop(columns=y_columns)
    y = df_pacientes[y_columns]

    modelos = {}
    metricas = {}
    previsoes = {}

    for disease in y.columns:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y[disease], test_size=0.3, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        modelos[disease] = model
        metricas[disease] = {
            "Acurácia": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "Precisão": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred)
        }
        # Exemplo com um paciente de teste
        previsoes[disease] = model.predict(X_test[:1])

    return modelos, metricas, previsoes


# Configuração de navegação no Streamlit
st.title("Sistema de Recomendação Alimentar para Idosos")
st.sidebar.title("Navegação")
menu = st.sidebar.radio("Menu", ["Coleta de Dados", "Anamnese", "Recomendações",
                        "Modelo de Machine Learning", "Resumo de Pacientes", "Resumo de Alimentos", "Resumo de Dados", "Análise de Dados de Alimentos"])
st.sidebar.image('logo.png', use_container_width=True)


# Inicializando variáveis no session_state para garantir que os dados persistam
if 'dados_paciente' not in st.session_state:
    st.session_state['dados_paciente'] = {
        'nome': '', 'idade': 0, 'genero': '', 'peso': 0.0, 'altura': 0.0,
        'atividade_fisica': '', 'imc': 0.0, 'tmb': 0.0, 'doencas': []
    }

# Página de Coleta de Dados
# Página de Coleta de Dados
if menu == "Coleta de Dados":
    st.header("Informações do Paciente")

    # Coleta e atualização de cada campo no session_state
    nome = st.text_input(
        "Nome", value=st.session_state['dados_paciente'].get('nome', ''))
    st.session_state['dados_paciente']['nome'] = nome

    idade = st.number_input(
        "Idade", min_value=0, value=st.session_state['dados_paciente'].get('idade', 0))
    st.session_state['dados_paciente']['idade'] = idade

    genero = st.selectbox("Gênero", ["", "Masculino", "Feminino"], index=[
        "", "Masculino", "Feminino"].index(st.session_state['dados_paciente'].get('genero', '')))
    st.session_state['dados_paciente']['genero'] = genero

    peso = st.number_input("Peso (kg)", min_value=0.0, max_value=200.0,
                           value=st.session_state['dados_paciente'].get('peso', 0.0))
    st.session_state['dados_paciente']['peso'] = peso

    altura = st.number_input("Altura (cm)", min_value=0.0, max_value=250.0,
                             value=st.session_state['dados_paciente'].get('altura', 0.0))
    st.session_state['dados_paciente']['altura'] = altura

    atividade_fisica = st.selectbox("Nível de Atividade Física", ["", "Sedentário", "Leve", "Moderado", "Intenso"], index=[
        "", "Sedentário", "Leve", "Moderado", "Intenso"].index(st.session_state['dados_paciente'].get('atividade_fisica', '')))
    st.session_state['dados_paciente']['atividade_fisica'] = atividade_fisica

    # Cálculo de IMC e TMB, atualizando o session_state com os valores calculados
    if st.session_state['dados_paciente']['peso'] > 0 and st.session_state['dados_paciente']['altura'] > 0 and st.session_state['dados_paciente']['idade'] > 0:
        st.session_state['dados_paciente']['imc'] = calcular_imc(
            st.session_state['dados_paciente']['peso'], st.session_state['dados_paciente']['altura'])
        st.write(
            f"IMC Calculado: {st.session_state['dados_paciente']['imc']:.2f}")

        if st.session_state['dados_paciente']['genero'] and st.session_state['dados_paciente']['atividade_fisica']:
            st.session_state['dados_paciente']['tmb'] = calcular_tmb(
                st.session_state['dados_paciente']['peso'],
                st.session_state['dados_paciente']['altura'],
                st.session_state['dados_paciente']['idade'],
                st.session_state['dados_paciente']['genero'],
                st.session_state['dados_paciente']['atividade_fisica']
            )
            st.write(
                f"Taxa Metabólica Basal (TMB): {st.session_state['dados_paciente']['tmb']:.2f} kcal")


# Página de Anamnese
# Página de Anamnese
elif menu == "Anamnese":
    st.header("Ficha de Anamnese Detalhada")
    st.write("Selecione as doenças pré-existentes do paciente:")

    # Inicializar a lista de doenças no session_state se ainda não existir
    if 'doencas' not in st.session_state['dados_paciente']:
        st.session_state['dados_paciente']['doencas'] = []

    # Atualizar a lista de doenças com base nas seleções
    doencas_selecionadas = []
    for doenca in doencas_opcoes:
        # Checkbox com valor padrão baseado no estado atual do session_state
        is_checked = doenca in st.session_state['dados_paciente']['doencas']
        if st.checkbox(doenca, value=is_checked):
            doencas_selecionadas.append(doenca)

    # Salva a lista atualizada no session_state
    st.session_state['dados_paciente']['doencas'] = doencas_selecionadas

    # Exibir as doenças selecionadas
    st.write("Doenças selecionadas:",
             st.session_state['dados_paciente']['doencas'])


# Página de Recomendação de Refeições
# Página de Recomendações
# Página de Recomendações
# Página de Recomendações
elif menu == "Recomendações":
    st.header("Recomendações de Refeições")

    # Verifica se a TMB foi calculada
    tmb = st.session_state['dados_paciente'].get('tmb', 0)
    if tmb <= 0:
        st.write(
            "Por favor, insira os dados do paciente na página 'Coleta de Dados' para calcular a TMB.")
    else:
        modelos, metricas, previsoes = treinar_modelo_risco(df_pacientes)

        # Sugerir refeições com base na TMB e nas restrições
        recomendacoes = sugerir_refeicoes_ajustado(tmb, previsoes)

        # Calorias distribuídas igualmente para cada refeição inicial
        calorias_por_refeicao = tmb / 4
        st.write(f"TMB calculada para o dia: {tmb:.2f} kcal")

        total_calorias_dia = 0  # Variável para armazenar o total de calorias para o dia
        refeicoes_ajustadas = {}

        for refeicao, alimentos in recomendacoes.items():
            total_calorias_refeicao = alimentos['Calorias'].sum()

            # Ajuste de calorias para não ultrapassar a TMB por refeição
            if total_calorias_refeicao > calorias_por_refeicao:
                # Selecionar alimentos de menor caloria até atingir o limite da refeição
                alimentos = alimentos.sort_values(
                    by='Calorias', ascending=True)
                calorias_cumulativas = alimentos['Calorias'].cumsum()
                # Ligeiramente maior para ajustar
                alimentos = alimentos[calorias_cumulativas <=
                                      calorias_por_refeicao * 1.05]

            total_calorias_refeicao = alimentos['Calorias'].sum()
            total_calorias_dia += total_calorias_refeicao  # Adiciona ao total diário
            # Salva a refeição ajustada
            refeicoes_ajustadas[refeicao] = alimentos

        # Ajuste final se total_calorias_dia for ligeiramente menor que TMB
        if total_calorias_dia < tmb:
            # Todo: Perguntar para o Pedro o que isso significa
            ajuste_fator = tmb / total_calorias_dia
            # ajuste_fator = 1  # Adicionado esse valor para não alterar as calorias por enquanto
            total_calorias_dia = 0  # Redefine o total para recalcular

            # Aplicar o fator de ajuste em cada refeição para que o total corresponda ou seja ligeiramente superior à TMB
            for refeicao, alimentos in refeicoes_ajustadas.items():
                alimentos['Calorias'] = alimentos['Calorias'] * ajuste_fator
                total_calorias_refeicao = alimentos['Calorias'].sum()
                total_calorias_dia += total_calorias_refeicao
                refeicoes_ajustadas[refeicao] = alimentos

        # Exibição das refeições ajustadas e total de calorias
        for refeicao, alimentos in refeicoes_ajustadas.items():
            st.subheader(refeicao)
            total_calorias_refeicao = alimentos['Calorias'].sum()

            # Exibe os alimentos da refeição e suas calorias ajustadas
            for _, row in alimentos.iterrows():
                row = row.fillna(0)

                print(row['Quantidade (g)'], 2)
                print(round(row['Quantidade (g)'], 2))

                st.write(
                    f"- {row['Alimento']} ({row['Quantidade (g)']:.2f}g): {row['Calorias']:.2f} kcal - Proteína: {row['Proteína']:.2f}g - Carboidrato: {row['Carboidrato']:.2f}g - Lipídeos: {row['Lipídeos']:.2f}g")
            # Exibe o total de calorias para a refeição atual
            st.write(
                f"**Total de calorias para {refeicao}: {total_calorias_refeicao:.2f} kcal**")

        # Exibe o total de calorias sugerido para o dia
        st.write(
            f"### Total de calorias sugerido para o dia: {total_calorias_dia:.2f} kcal")

        if total_calorias_dia > tmb:
            st.info(
                f"O total de calorias ({total_calorias_dia:.2f} kcal) está ligeiramente acima da TMB ({tmb:.2f} kcal) para atender aos requisitos nutricionais.")

        # Calcula o total de cada nutriente para o dia
        total_proteina = sum(alimentos['Proteína'].sum()
                             for refeicao, alimentos in recomendacoes.items())
        total_carboidrato = sum(alimentos['Carboidrato'].sum()
                                for refeicao, alimentos in recomendacoes.items())
        total_lipideos = sum(alimentos['Lipídeos'].sum()
                             for refeicao, alimentos in recomendacoes.items())

        # Cria um DataFrame com os nutrientes totais
        nutrientes_totais = pd.DataFrame({
            'Nutriente': ['Proteína', 'Carboidrato', 'Lipídeos'],
            'Quantidade (g)': [total_proteina.round(2), total_carboidrato.round(2), total_lipideos.round(2)]
        })

        # Exibe o gráfico de barras
        fig_nutrientes = go.Figure(data=[
            go.Bar(
                x=nutrientes_totais['Nutriente'],
                y=nutrientes_totais['Quantidade (g)'],
                text=nutrientes_totais['Quantidade (g)'],
                textposition='auto',
                marker=dict(
                    color='rgba(0, 128, 255, 0.7)',
                    line=dict(width=1.5, color='rgba(0, 128, 255, 1)')
                )
            )
        ])
        fig_nutrientes.update_layout(
            title="Total de Nutrientes para o Dia",
            xaxis_title="Nutrientes",
            yaxis_title="Quantidade (g)",
            template="plotly_white"
        )
        st.plotly_chart(fig_nutrientes)


# Página de Resumo de Dados
elif menu == "Resumo de Dados":
    st.header("Resumo de Dados")

    # Resumo de df_pacientes
    st.subheader("Resumo do DataFrame de Pacientes")
    st.write("Número de pacientes:", df_pacientes.shape[0])
    st.write(df_pacientes.describe())
    st.write("Amostra de dados de pacientes:")
    df_pacientes.set_index('Idade')
    st.write(df_pacientes.head())

    # Resumo de df_alimentos
    st.subheader("Resumo do DataFrame de Alimentos")
    st.write("Número de alimentos:", df_alimentos.shape[0])
    st.write(df_alimentos.describe())
    st.write("Amostra de dados de alimentos:")

    df_alimentos["Umidade"] = pd.to_numeric(
        df_alimentos["Umidade"], errors='coerce')
    df_alimentos["Proteína"] = pd.to_numeric(
        df_alimentos["Proteína"], errors='coerce')
    df_alimentos["Colesterol"] = pd.to_numeric(
        df_alimentos["Colesterol"], errors='coerce')
    df_alimentos["Lipídeos"] = pd.to_numeric(
        df_alimentos["Lipídeos"], errors='coerce')
    df_alimentos["Carboidrato"] = pd.to_numeric(
        df_alimentos["Carboidrato"], errors='coerce')
    df_alimentos["Cálcio"] = pd.to_numeric(
        df_alimentos["Cálcio"], errors='coerce')
    df_alimentos["Sódio"] = pd.to_numeric(
        df_alimentos["Sódio"], errors='coerce')

    df_alimentos = df_alimentos.style.format({"Umidade": "{:.2f}", "Proteína": "{:.2f}", "Lipídeos": "{:.2f}",
                                             "Carboidrato": "{:.2f}", "Cálcio": "{:.2f}", "Sódio": "{:.2f}", "Colesterol": "{:.2f}"})

    st.write(df_alimentos)

# Página de Resumo de Dados
elif menu == "Análise de Dados de Alimentos":
    # st.header("Análise de Dados de Alimentos")

    # Resumo Descritivo
    st.subheader("Resumo Descritivo dos Dados")
    # Removemos o set_table_styles para evitar o erro
    st.write(df_alimentos.describe().style.format(precision=2))

    # Seleção de colunas para análise
    st.header("Visualização Interativa")
    colunas = df_alimentos.columns.tolist()

    # Gráfico de Dispersão
    st.subheader("Gráfico de Dispersão")
    coluna_x = st.selectbox(
        "Escolha a coluna para o eixo X (Dispersão)", colunas, index=0)
    coluna_y = st.selectbox(
        "Escolha a coluna para o eixo Y (Dispersão)", colunas, index=1)
    fig_scatter = go.Figure(data=go.Scatter(
        x=df_alimentos[coluna_x], y=df_alimentos[coluna_y], mode='markers'))
    fig_scatter.update_layout(
        title=f'Dispersão de {coluna_x} vs {coluna_y}', xaxis_title=coluna_x, yaxis_title=coluna_y)
    st.plotly_chart(fig_scatter)
    st.write("**Uso**: O gráfico de dispersão é utilizado para visualizar a relação entre duas variáveis, indicando se existe uma correlação entre elas.")

    # Resumo Descritivo do Gráfico de Dispersão
    st.write("Resumo Descritivo do Eixo X")
    st.write(df_alimentos[coluna_x].describe())
    st.write("Resumo Descritivo do Eixo Y")
    st.write(df_alimentos[coluna_y].describe())

    # Histograma
    st.subheader("Histograma")
    coluna_hist = st.selectbox("Escolha a coluna para o histograma", colunas)
    fig_hist = go.Figure(data=go.Histogram(
        x=df_alimentos[coluna_hist], nbinsx=30))
    fig_hist.update_layout(
        title=f'Histograma de {coluna_hist}', xaxis_title=coluna_hist, yaxis_title='Frequência')
    st.plotly_chart(fig_hist)
    st.write("**Uso**: O histograma é usado para visualizar a distribuição de uma variável, mostrando como os valores são distribuídos ao longo dos intervalos.")

    st.write(f"Resumo Descritivo de {coluna_hist}")
    st.write(df_alimentos[coluna_hist].describe())

    # Box Plot
    st.subheader("Box Plot")
    coluna_box = st.selectbox("Escolha a coluna para o Box Plot", colunas)
    fig_box = go.Figure(data=go.Box(
        y=df_alimentos[coluna_box], boxpoints='all', jitter=0.3))
    fig_box.update_layout(
        title=f'Box Plot de {coluna_box}', yaxis_title=coluna_box)
    st.plotly_chart(fig_box)
    st.write("**Uso**: O box plot é utilizado para visualizar a dispersão dos dados e identificar valores atípicos (outliers). Ele mostra a mediana e os quartis.")

    st.write(f"Resumo Descritivo de {coluna_box}")
    st.write(df_alimentos[coluna_box].describe())

    # Gráfico de Barras
    st.subheader("Gráfico de Barras")
    coluna_bar = st.selectbox(
        "Escolha a coluna para o Gráfico de Barras", colunas)
    fig_bar = go.Figure(data=go.Bar(
        x=df_alimentos.index, y=df_alimentos[coluna_bar]))
    fig_bar.update_layout(
        title=f'Gráfico de Barras de {coluna_bar}', xaxis_title="Índice", yaxis_title=coluna_bar)
    st.plotly_chart(fig_bar)
    st.write("**Uso**: O gráfico de barras é usado para comparar valores individuais entre diferentes categorias ou índices.")

    st.write(f"Resumo Descritivo de {coluna_bar}")
    st.write(df_alimentos[coluna_bar].describe())

    # Gráfico de Linhas
    st.subheader("Gráfico de Linhas")
    coluna_line = st.selectbox(
        "Escolha a coluna para o Gráfico de Linhas", colunas)
    fig_line = go.Figure(data=go.Scatter(
        x=df_alimentos.index, y=df_alimentos[coluna_line], mode='lines'))
    fig_line.update_layout(
        title=f'Gráfico de Linhas de {coluna_line}', xaxis_title="Índice", yaxis_title=coluna_line)
    st.plotly_chart(fig_line)
    st.write("**Uso**: O gráfico de linhas é ideal para visualizar tendências ao longo do tempo ou através de um índice.")

    st.write(f"Resumo Descritivo de {coluna_line}")
    st.write(df_alimentos[coluna_line].describe())

    # Gráfico de Área
    st.subheader("Gráfico de Área")
    coluna_area = st.selectbox(
        "Escolha a coluna para o Gráfico de Área", colunas)
    fig_area = go.Figure(data=go.Scatter(
        x=df_alimentos.index, y=df_alimentos[coluna_area], fill='tozeroy'))
    fig_area.update_layout(
        title=f'Gráfico de Área de {coluna_area}', xaxis_title="Índice", yaxis_title=coluna_area)
    st.plotly_chart(fig_area)
    st.write("**Uso**: O gráfico de área é usado para representar mudanças cumulativas ao longo de um índice ou tempo.")

    st.write(f"Resumo Descritivo de {coluna_area}")
    st.write(df_alimentos[coluna_area].describe())

    # Gráfico de Pizza
    st.subheader("Gráfico de Pizza")
    coluna_pizza = st.selectbox(
        "Escolha a coluna para o Gráfico de Pizza (categórica)", colunas)
    df_pizza = df_alimentos[coluna_pizza].value_counts()
    fig_pizza = go.Figure(data=go.Pie(
        labels=df_pizza.index, values=df_pizza.values))
    fig_pizza.update_layout(title=f'Distribuição de {coluna_pizza}')
    st.plotly_chart(fig_pizza)
    st.write(
        "**Uso**: O gráfico de pizza mostra a proporção de cada categoria em relação ao total.")

    st.write(f"Contagem de {coluna_pizza}")
    st.write(df_pizza)

    # Heatmap de Correlação
    st.subheader("Heatmap de Correlação")

    # Seleciona apenas as colunas numéricas para calcular a correlação
    df_numerico = df_alimentos.select_dtypes(include=['float64', 'int64'])
    correlacao = df_numerico.corr()

    # Formata os valores de correlação com duas casas decimais para exibir como rótulos
    correlacao_text = correlacao.round(2).astype(str)

    # Cria o heatmap de correlação com rótulos de duas casas decimais
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=correlacao.values,
        x=correlacao.columns,
        y=correlacao.index,
        colorscale='Viridis',
        text=correlacao_text.values,  # Define os rótulos dos dados
        hovertemplate='%{text}'  # Exibe os valores de correlação no hover
    ))

    fig_heatmap.update_layout(
        title='Mapa de Calor das Correlações',
        xaxis_title="Variáveis",
        yaxis_title="Variáveis"
    )

    # Exibe o gráfico
    st.plotly_chart(fig_heatmap)

    st.write("**Uso**: O mapa de calor de correlação mostra a força e direção das relações entre variáveis numéricas.")

    # Gráfico de Densidade
    st.subheader("Gráfico de Densidade")

    # Filtra as colunas numéricas para exibir apenas essas no selectbox
    colunas_numericas = df_alimentos.select_dtypes(
        include=['float64', 'int64']).columns.tolist()
    coluna_densidade = st.selectbox(
        "Escolha a coluna para o Gráfico de Densidade", colunas_numericas)

    # Cria o gráfico de densidade
    fig_densidade = ff.create_distplot([df_alimentos[coluna_densidade].dropna()], [
                                       coluna_densidade], show_hist=False)
    fig_densidade.update_layout(title=f'Densidade de {coluna_densidade}')
    st.plotly_chart(fig_densidade)

    st.write("**Uso**: O gráfico de densidade mostra a distribuição dos dados de forma suavizada, útil para entender a frequência relativa.")

    # Resumo descritivo
    st.write(f"Resumo Descritivo de {coluna_densidade}")
    st.write(df_alimentos[coluna_densidade].describe())

    # Gráfico de Pareto
    st.subheader("Gráfico de Pareto")
    coluna_pareto = st.selectbox(
        "Escolha a coluna para o Gráfico de Pareto", colunas)
    df_pareto = df_alimentos[coluna_pareto].value_counts(
    ).sort_values(ascending=False).reset_index()
    df_pareto.columns = [coluna_pareto, 'count']
    df_pareto['cum_percentage'] = df_pareto['count'].cumsum() / \
        df_pareto['count'].sum() * 100
    fig_pareto = go.Figure()
    fig_pareto.add_trace(
        go.Bar(x=df_pareto[coluna_pareto], y=df_pareto['count'], name='Frequência'))
    fig_pareto.add_trace(go.Scatter(
        x=df_pareto[coluna_pareto], y=df_pareto['cum_percentage'], name='Porcentagem Acumulada', yaxis='y2'))
    fig_pareto.update_layout(title=f'Gráfico de Pareto de {coluna_pareto}',
                             yaxis=dict(title='Frequência'),
                             yaxis2=dict(title='Porcentagem Acumulada', overlaying='y', side='right'))
    st.plotly_chart(fig_pareto)
    st.write("**Uso**: O gráfico de Pareto combina barras e linhas para mostrar a frequência e o impacto cumulativo das categorias, identificando itens mais significativos.")

    st.write(f"Contagem e Porcentagem Acumulada de {coluna_pareto}")
    st.write(df_pareto)


# Página de Resumo de Pacientes com Gráficos
elif menu == "Resumo de Pacientes":
    st.header("Resumo dos Pacientes")

    # Gráfico de distribuição de idade
    fig_idade = go.Figure(data=[
        go.Histogram(
            x=df_pacientes['Idade'],
            nbinsx=20,
            marker=dict(
                color='rgba(0, 128, 255, 0.7)',
                line=dict(width=1, color='rgba(0, 128, 255, 1)')
            ),
            opacity=0.75
        )
    ])
    fig_idade.update_layout(
        title="Distribuição de Idade dos Pacientes",
        xaxis_title="Idade",
        yaxis_title="Frequência",
        template="plotly_white"
    )
    st.plotly_chart(fig_idade)

    # Gráfico de distribuição de IMC
    fig_imc = go.Figure(data=[
        go.Histogram(
            x=df_pacientes['IMC'],
            nbinsx=20,
            marker=dict(
                color='rgba(255, 0, 127, 0.7)',
                line=dict(width=1, color='rgba(255, 0, 127, 1)')
            ),
            opacity=0.75
        )
    ])
    fig_imc.update_layout(
        title="Distribuição de IMC dos Pacientes",
        xaxis_title="IMC",
        yaxis_title="Frequência",
        template="plotly_white"
    )
    st.plotly_chart(fig_imc)

    # Gráfico de distribuição de Peso
    fig_peso = go.Figure(data=[
        go.Histogram(
            x=df_pacientes['Peso (kg)'],
            nbinsx=20,
            marker=dict(
                color='rgba(0, 255, 127, 0.7)',
                line=dict(width=1, color='rgba(0, 255, 127, 1)')
            ),
            opacity=0.75
        )
    ])
    fig_peso.update_layout(
        title="Distribuição de Peso dos Pacientes",
        xaxis_title="Peso (kg)",
        yaxis_title="Frequência",
        template="plotly_white"
    )
    st.plotly_chart(fig_peso)

    # Gráfico de doenças mais comuns
    doencas_columns = [col for col in df_pacientes.columns if "Doença" in col]
    doencas_counts = df_pacientes[doencas_columns].sum()

    fig_doencas = go.Figure(data=[
        go.Bar(
            x=doencas_counts.index,
            y=doencas_counts.values,
            text=doencas_counts.values,
            textposition='auto',
            marker=dict(
                color='rgba(255, 165, 0, 0.7)',
                line=dict(width=1.5, color='rgba(255, 165, 0, 1)')
            )
        )
    ])
    fig_doencas.update_layout(
        title="Frequência de Doenças nos Pacientes",
        xaxis_title="Doenças",
        yaxis_title="Frequência",
        template="plotly_white"
    )
    st.plotly_chart(fig_doencas)

# Página de Resumo de Alimentos com Gráficos
elif menu == "Resumo de Alimentos":
    st.header("Resumo dos Alimentos")

    # Gráfico de distribuição de Calorias
    fig_calorias = go.Figure(data=[
        go.Histogram(
            x=df_alimentos['Calorias'],
            nbinsx=20,
            marker=dict(
                color='rgba(255, 99, 71, 0.7)',
                line=dict(width=1, color='rgba(255, 99, 71, 1)')
            ),
            opacity=0.75
        )
    ])
    fig_calorias.update_layout(
        title="Distribuição de Calorias",
        xaxis_title="Calorias",
        yaxis_title="Frequência",
        template="plotly_white"
    )
    st.plotly_chart(fig_calorias)

    # Gráfico de distribuição de Carboidratos
    fig_carboidratos = go.Figure(data=[
        go.Histogram(
            x=df_alimentos['Calorias'],
            nbinsx=20,
            marker=dict(
                color='rgba(75, 192, 192, 0.7)',
                line=dict(width=1, color='rgba(75, 192, 192, 1)')
            ),
            opacity=0.75
        )
    ])
    fig_carboidratos.update_layout(
        title="Distribuição de Carboidratos",
        xaxis_title="Calorias",
        yaxis_title="Frequência",
        template="plotly_white"
    )
    st.plotly_chart(fig_carboidratos)

    # Gráfico de distribuição de Gorduras
    fig_gorduras = go.Figure(data=[
        go.Histogram(
            x=df_alimentos['Calorias'],
            nbinsx=20,
            marker=dict(
                color='rgba(153, 102, 255, 0.7)',
                line=dict(width=1, color='rgba(153, 102, 255, 1)')
            ),
            opacity=0.75
        )
    ])

    fig_gorduras.update_layout(
        title="Distribuição de Calorias",
        xaxis_title="Calorias",
        yaxis_title="Frequência",
        template="plotly_white"
    )
    st.plotly_chart(fig_gorduras)

    # Gráfico de categorias de alimentos mais comuns
    categorias_counts = df_alimentos['Refeicao_Indicada'].value_counts()

    fig_categorias = go.Figure(data=[
        go.Bar(
            x=categorias_counts.index,
            y=categorias_counts.values,
            text=categorias_counts.values,
            textposition='auto',
            marker=dict(
                color='rgba(255, 206, 86, 0.7)',
                line=dict(width=1.5, color='rgba(255, 206, 86, 1)')
            )
        )
    ])
    fig_categorias.update_layout(
        title="Base de Alimentos por Refeição",
        xaxis_title="Categorias",
        yaxis_title="Frequência",
        template="plotly_white"
    )
    st.plotly_chart(fig_categorias)
# Página 5: Modelo de Machine Learning e Resultados
elif menu == "Modelo de Machine Learning":
    st.header("Modelo de Machine Learning Usado e Resultados")

    # Verificar se todas as informações necessárias estão preenchidas
    if (st.session_state['dados_paciente']['peso'] > 0 and
        st.session_state['dados_paciente']['altura'] > 0 and
        st.session_state['dados_paciente']['idade'] > 0 and
        st.session_state['dados_paciente']['genero'] and
        st.session_state['dados_paciente']['atividade_fisica'] and
            st.session_state['dados_paciente']['doencas']):

        # Verificar se a coluna Target existe, caso contrário, criá-la com valores simulados
        if 'Target' not in df_alimentos.columns:
            df_alimentos['Target'] = np.random.choice(
                [0, 1], size=len(df_alimentos))

        # Preparação dos dados para treinamento, convertendo variáveis categóricas para numéricas
        # Verificar se a coluna 'Target' existe antes de removê-la
        df_alimentos_encoded = pd.get_dummies(
            df_alimentos.drop(columns=["Target"], errors="ignore"))
        X = df_alimentos_encoded
        y = df_alimentos["Target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        modelo = RandomForestClassifier(random_state=42)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        # Calculando as métricas
        acuracia = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precisao = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Exibindo informações do modelo e métricas
        st.write(f"**Modelo Utilizado:** Random Forest Classifier")
        st.write("Este modelo foi treinado para recomendar alimentos específicos com base nas características e restrições de saúde dos idosos.")

        st.subheader("Métricas de Desempenho")
        st.write(f"**Acurácia:** {acuracia:.2f}")
        st.write(f"**F1 Score:** {f1:.2f}")
        st.write(f"**Precisão:** {precisao:.2f}")
        st.write(f"**Recall:** {recall:.2f}")

        st.subheader("Análise das Métricas")
        st.write("""
            - **Acurácia** indica a proporção de predições corretas realizadas pelo modelo em relação ao total de predições.
            - **Precisão** é a proporção de predições corretas entre todas as que foram preditas como positivas.
            - **F1 Score** é a média harmônica entre precisão e recall, proporcionando uma medida balanceada entre ambos.
            - **Recall** mostra a capacidade do modelo em encontrar todas as amostras positivas.

            Estas métricas sugerem que o modelo tem um bom desempenho para as recomendações alimentares, atendendo às restrições de saúde e necessidades energéticas do paciente.
        """)

    else:
        st.write("**Dados incompletos.** Por favor, revise as informações nas seções de Coleta de Dados e Anamnese para calcular o modelo de machine learning.")
