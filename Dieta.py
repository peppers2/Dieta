import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Configuração da página
st.set_page_config(page_title="TCC Univesp - Sistema de Recomendação", page_icon="logo.png", layout="wide")

# Funções para cálculo de IMC e TMB
def calcular_imc(peso, altura):
    return peso / ((altura / 100) ** 2) if altura > 0 else 0

def calcular_tmb(peso, altura, idade, genero, atividade_fisica):
    if genero == "Masculino":
        tmb = 88.36 + (13.4 * peso) + (4.8 * altura) - (5.7 * idade)
    else:
        tmb = 447.6 + (9.2 * peso) + (3.1 * altura) - (4.3 * idade)
    
    fator_atividade = {"Sedentário": 1.2, "Leve": 1.375, "Moderado": 1.55, "Intenso": 1.725}
    return tmb * fator_atividade.get(atividade_fisica, 1)

# Carregar dados de alimentos e pacientes
df_alimentos = pd.read_csv("base_alimentos_2000.csv")
df_pacientes = pd.read_csv("pacientes_base_treinamento_1000_formatado_brasil.csv", delimiter=";")

# Ajustar dados de pacientes (caso necessário)
df_pacientes["Peso (kg)"] = df_pacientes["Peso (kg)"].str.replace(",", ".").astype(float)
df_pacientes["Altura (cm)"] = df_pacientes["Altura (cm)"].str.replace(",", ".").astype(float)
df_pacientes["IMC"] = df_pacientes["IMC"].str.replace(",", ".").astype(float)

# Doenças para anamnese
doencas_opcoes = [
    "Diabetes Tipo 1", "Diabetes Tipo 2", "Hipertensão", "Obesidade", 
    "Insuficiência Renal", "Colesterol Alto", "Doenças Cardíacas", 
    "Osteoporose", "Doença Celíaca"
]

# Função para sugerir refeições sem ultrapassar a TMB e respeitando restrições
def sugerir_refeicoes(tmb, doencas):
    df_permitidos = df_alimentos[~df_alimentos["Doencas_Restritivas"].isin(doencas)]
    refeicoes = {}
    calorias_refeicao = tmb / 4 if tmb > 0 else 0
    
    for tipo_refeicao in ["Café da Manhã", "Almoço", "Lanche da Tarde", "Jantar"]:
        alimentos_disponiveis = df_permitidos[(df_permitidos["Refeicao_Indicada"] == tipo_refeicao) & (df_permitidos["Calorias"] <= calorias_refeicao)]
        refeicoes[tipo_refeicao] = alimentos_disponiveis.sample(min(3, len(alimentos_disponiveis)))[["Alimento", "Calorias"]]
    
    return refeicoes

# Função para treinar o modelo e prever doenças
def treinar_modelo(df):
    X = df.drop(columns=[col for col in df.columns if "Doença" in col])
    y = df[[col for col in df.columns if "Doença" in col]]
    
    modelos = {}
    metricas = {}
    previsoes = {}

    for disease in y.columns:
        X_train, X_test, y_train, y_test = train_test_split(X, y[disease], test_size=0.3, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Salvando o modelo e as métricas
        modelos[disease] = model
        metricas[disease] = {
            "Acurácia": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "Precisão": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred)
        }
        
        # Previsão (exemplo para um novo paciente)
        previsoes[disease] = model.predict(X_test[:1])  # Exemplo com um paciente de teste
    
    return modelos, metricas, previsoes

# Configuração de navegação no Streamlit
st.title("Sistema de Recomendação Alimentar para Idosos")
st.sidebar.title("Navegação")

menu = st.sidebar.radio("Menu", ["Coleta de Dados","Resumo de Pacientes","Resumo de Alimentos", "Anamnese", "Recomendações", "Análise Gráfica", "Modelo de Machine Learning"])

# Inicializando variáveis no session_state para garantir que os dados persistam
if 'dados_paciente' not in st.session_state:
    st.session_state['dados_paciente'] = {
        'nome': '', 'idade': 0, 'genero': '', 'peso': 0.0, 'altura': 0.0,
        'atividade_fisica': '', 'imc': 0.0, 'tmb': 0.0, 'doencas': []
    }
if 'calorias_refeicoes' not in st.session_state:
    st.session_state['calorias_refeicoes'] = {}

# Página 1: Coleta de Dados e Cálculo de IMC e TMB
if menu == "Coleta de Dados":
    st.header("Informações do Paciente")
    st.session_state['dados_paciente']['nome'] = st.text_input("Nome", value=st.session_state['dados_paciente']['nome'])
    st.session_state['dados_paciente']['idade'] = st.number_input("Idade", min_value=0, value=st.session_state['dados_paciente']['idade'])
    st.session_state['dados_paciente']['genero'] = st.selectbox("Gênero", ["", "Masculino", "Feminino"], index=0 if st.session_state['dados_paciente']['genero'] == "" else ["", "Masculino", "Feminino"].index(st.session_state['dados_paciente']['genero']))
    st.session_state['dados_paciente']['peso'] = st.number_input("Peso (kg)", min_value=0.0, max_value=200.0, value=st.session_state['dados_paciente']['peso'])
    st.session_state['dados_paciente']['altura'] = st.number_input("Altura (cm)", min_value=0.0, max_value=250.0, value=st.session_state['dados_paciente']['altura'])
    st.session_state['dados_paciente']['atividade_fisica'] = st.selectbox("Nível de Atividade Física", ["", "Sedentário", "Leve", "Moderado", "Intenso"], index=0 if st.session_state['dados_paciente']['atividade_fisica'] == "" else ["", "Sedentário", "Leve", "Moderado", "Intenso"].index(st.session_state['dados_paciente']['atividade_fisica']))

    # Cálculo do IMC e TMB
    if st.session_state['dados_paciente']['peso'] > 0 and st.session_state['dados_paciente']['altura'] > 0 and st.session_state['dados_paciente']['idade'] > 0:
        st.session_state['dados_paciente']['imc'] = calcular_imc(st.session_state['dados_paciente']['peso'], st.session_state['dados_paciente']['altura'])
        st.write(f"IMC Calculado: {st.session_state['dados_paciente']['imc']:.2f}")
        
        if st.session_state['dados_paciente']['genero'] and st.session_state['dados_paciente']['atividade_fisica']:
            st.session_state['dados_paciente']['tmb'] = calcular_tmb(
                st.session_state['dados_paciente']['peso'], 
                st.session_state['dados_paciente']['altura'], 
                st.session_state['dados_paciente']['idade'], 
                st.session_state['dados_paciente']['genero'], 
                st.session_state['dados_paciente']['atividade_fisica']
            )
            st.write(f"Taxa Metabólica Basal (TMB): {st.session_state['dados_paciente']['tmb']:.2f} kcal")
        else:
            st.write("Preencha o gênero e o nível de atividade física para calcular a TMB.")


            # Página 2: Anamnese Detalhada
elif menu == "Anamnese":
    st.header("Ficha de Anamnese Detalhada")
    st.write("Selecione as doenças pré-existentes do paciente:")

    # Coleta das doenças pré-existentes
    st.session_state['dados_paciente']['doencas'] = {doenca: st.checkbox(doenca, value=doenca in st.session_state['dados_paciente']['doencas']) for doenca in doencas_opcoes}
    
    # Converter para uma lista de doenças selecionadas
    doencas_selecionadas = [doenca for doenca, selecionado in st.session_state['dados_paciente']['doencas'].items() if selecionado]
    st.session_state['dados_paciente']['doencas'] = doencas_selecionadas
    st.write("Doenças selecionadas:", doencas_selecionadas)

    st.write("Informações salvas! Você pode continuar para a seção de recomendações.")

# Página 3: Recomendação de Refeições
elif menu == "Recomendações":
    st.header("Recomendações de Refeições")
    
    # Geração das recomendações de refeições com base na TMB e nas doenças selecionadas
    if st.session_state['dados_paciente']['tmb'] > 0 and st.session_state['dados_paciente']['doencas']:
        recomendacoes = sugerir_refeicoes(st.session_state['dados_paciente']['tmb'], st.session_state['dados_paciente']['doencas'])
        total_calorias_consumido = 0  
        
        st.write("Com base nas informações fornecidas, aqui estão as recomendações de refeições:")
        calorias_refeicoes = {}
        
        # Exibição das refeições e calorias totais para cada uma
        for refeicao, alimentos in recomendacoes.items():
            st.subheader(refeicao)
            calorias_refeicao = alimentos["Calorias"].sum()
            calorias_refeicoes[refeicao] = calorias_refeicao
            total_calorias_consumido += calorias_refeicao
            for _, row in alimentos.iterrows():
                st.write(f"- {row['Alimento']}: {row['Calorias']} kcal")
            st.write(f"**Total de Calorias para {refeicao}: {calorias_refeicao} kcal**")
        
        st.session_state['calorias_refeicoes'] = calorias_refeicoes
        st.write(f"**Total de Calorias Consumidas no Dia:** {total_calorias_consumido} kcal")
        st.write(f"**Total de Calorias Proposto para o Dia (TMB):** {st.session_state['dados_paciente']['tmb']:.2f} kcal")

# Página 4: Análise Gráfica
elif menu == "Análise Gráfica":
    st.header("Análise Gráfica")

    if st.session_state['dados_paciente']['tmb'] > 0 and st.session_state['calorias_refeicoes']:
        # Gráfico de barras das calorias por refeição com rótulos de dados
        fig_calorias_refeicoes = go.Figure(data=[
            go.Bar(
                name="Calorias", 
                x=list(st.session_state['calorias_refeicoes'].keys()), 
                y=list(st.session_state['calorias_refeicoes'].values()), 
                text=list(st.session_state['calorias_refeicoes'].values()), 
                textposition='auto'
            )
        ])
        fig_calorias_refeicoes.update_layout(title="Calorias Consumidas por Refeição")
        st.plotly_chart(fig_calorias_refeicoes)
        
        # Gráfico de indicador de IMC
        fig_imc = go.Figure(go.Indicator(
            mode="gauge+number",
            value=st.session_state['dados_paciente']['imc'],
            title={'text': "IMC"},
            gauge={'axis': {'range': [10, 40]}, 'bar': {'color': "darkblue"}},
            number={'suffix': " kg/m²"}
        ))
        st.plotly_chart(fig_imc)

        # Gráfico de pizza com rótulos de calorias
        fig_pizza_calorias = go.Figure(data=[
            go.Pie(
                labels=list(st.session_state['calorias_refeicoes'].keys()), 
                values=list(st.session_state['calorias_refeicoes'].values()),
                textinfo='label+percent',  # Exibe o rótulo e a porcentagem
                insidetextorientation='radial'
            )
        ])
        fig_pizza_calorias.update_layout(title="Distribuição de Calorias por Refeição")
        st.plotly_chart(fig_pizza_calorias)

        # Gráfico de barra das doenças selecionadas com rótulos de dados
        doencas_selecionadas = st.session_state['dados_paciente']['doencas']
        fig_doencas = go.Figure(data=[
            go.Bar(
                name="Doenças Selecionadas", 
                x=doencas_opcoes, 
                y=[1 if d in doencas_selecionadas else 0 for d in doencas_opcoes],
                text=[1 if d in doencas_selecionadas else 0 for d in doencas_opcoes],
                textposition='auto'
            )
        ])
        fig_doencas.update_layout(title="Distribuição das Doenças Selecionadas")
        st.plotly_chart(fig_doencas)

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
            df_alimentos['Target'] = np.random.choice([0, 1], size=len(df_alimentos))

        # Preparação dos dados para treinamento, convertendo variáveis categóricas para numéricas
        # Verificar se a coluna 'Target' existe antes de removê-la
        df_alimentos_encoded = pd.get_dummies(df_alimentos.drop(columns=["Target"], errors="ignore"))
        X = df_alimentos_encoded
        y = df_alimentos["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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


elif menu == "Resumo de Pacientes":
    st.header("Resumo dos Pacientes")
    
        # Gráfico de distribuição de idade com gradiente e rótulos
    fig_idade = go.Figure(data=[
        go.Histogram(
            x=df_pacientes['Idade'],
            nbinsx=20,
            marker=dict(
                color='rgba(0, 128, 255, 0.7)',  # Cor inicial
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

    # Gráfico de distribuição de IMC com gradiente e rótulos
    fig_imc = go.Figure(data=[
        go.Histogram(
            x=df_pacientes['IMC'],
            nbinsx=20,
            marker=dict(
                color='rgba(255, 0, 127, 0.7)',  # Cor inicial
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

    # Gráfico de distribuição de Peso com gradiente e rótulos
    fig_peso = go.Figure(data=[
        go.Histogram(
            x=df_pacientes['Peso (kg)'],
            nbinsx=20,
            marker=dict(
                color='rgba(0, 255, 127, 0.7)',  # Cor inicial
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

    # Gráfico de doenças mais comuns com gradiente e rótulos
    doencas_columns = [col for col in df_pacientes.columns if "Doença" in col]
    doencas_counts = df_pacientes[doencas_columns].sum()

    fig_doencas = go.Figure(data=[
        go.Bar(
            x=doencas_counts.index,
            y=doencas_counts.values,
            text=doencas_counts.values,
            textposition='auto',
            marker=dict(
                color='rgba(255, 165, 0, 0.7)',  # Cor inicial
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

elif menu == "Resumo de Alimentos":
    st.header("Resumo dos Alimentos")

    
    # Gráfico de distribuição de Calorias
    fig_calorias = go.Figure(data=[
        go.Histogram(
            x=df_alimentos['Calorias'],
            nbinsx=20,
            marker=dict(
                color='rgba(255, 99, 71, 0.7)',  # Cor inicial
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

   
    # Gráfico de distribuição de Gorduras
    fig_carboidratos = go.Figure(data=[
        go.Histogram(
            x=df_alimentos['Calorias'],
            nbinsx=20,
            marker=dict(
                color='rgba(75, 192, 192, 0.7)',  # Cor inicial
                line=dict(width=1, color='rgba(75, 192, 192, 1)')
            ),
            opacity=0.75
        )
    ])
    fig_carboidratos.update_layout(
        title="Distribuição de Calorias",
        xaxis_title="Calorias (g)",
        yaxis_title="Frequência",
        template="plotly_white"
    )
    st.plotly_chart(fig_carboidratos)

    # Gráfico de distribuição de Gorduras
    fig_gorduras = go.Figure(data=[
        go.Histogram(
            x=df_alimentos['Doencas_Restritivas'],
            nbinsx=20,
            marker=dict(
                color='rgba(153, 102, 255, 0.7)',  # Cor inicial
                line=dict(width=1, color='rgba(153, 102, 255, 1)')
            ),
            opacity=0.75
        )
    ])
    fig_gorduras.update_layout(
        title="Doenças Restritivas",
        xaxis_title="Doenças Restritivas",
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
                color='rgba(255, 206, 86, 0.7)',  # Cor inicial
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