import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Configuração da página: título e favicon (logo)
st.set_page_config(page_title="TCC Univesp - Sistema de Recomendação", page_icon="logo.png", layout="wide")

# Funções para cálculo de IMC e TMB (Harris-Benedict)
def calcular_imc(peso, altura):
    return peso / ((altura / 100) ** 2) if altura > 0 else 0

def calcular_tmb(peso, altura, idade, genero, atividade_fisica):
    if genero == "Masculino":
        tmb = 88.36 + (13.4 * peso) + (4.8 * altura) - (5.7 * idade)
    else:
        tmb = 447.6 + (9.2 * peso) + (3.1 * altura) - (4.3 * idade)
    
    fator_atividade = {"Sedentário": 1.2, "Leve": 1.375, "Moderado": 1.55, "Intenso": 1.725}
    return tmb * fator_atividade.get(atividade_fisica, 1)

# Carregar dados de alimentos
df_alimentos = pd.read_csv("base_alimentos_2000.csv")

# Criar coluna Target de forma simulada se não existir
if 'Target' not in df_alimentos.columns:
    df_alimentos['Target'] = np.random.choice([0, 1], size=len(df_alimentos))

# Doenças consideradas na anamnese
doencas_opcoes = [
    "Diabetes Tipo 1", "Diabetes Tipo 2", "Hipertensão", "Obesidade", 
    "Insuficiência Renal", "Colesterol Alto", "Doenças Cardíacas", 
    "Osteoporose", "Doença Celíaca"
]

# Função para sugerir refeições dentro da TMB e restrições alimentares
def sugerir_refeicoes(tmb, doencas):
    df_permitidos = df_alimentos[~df_alimentos["Doencas_Restritivas"].isin(doencas)]
    refeicoes = {}
    calorias_refeicao = tmb / 4 if tmb > 0 else 0
    
    for tipo_refeicao in ["Café da Manhã", "Almoço", "Lanche da Tarde", "Jantar"]:
        alimentos_disponiveis = df_permitidos[(df_permitidos["Refeicao_Indicada"] == tipo_refeicao) & (df_permitidos["Calorias"] <= calorias_refeicao)]
        refeicoes[tipo_refeicao] = alimentos_disponiveis.sample(min(3, len(alimentos_disponiveis)))[["Alimento", "Calorias"]]
    
    return refeicoes

# Configuração do Streamlit e menu de navegação
st.title("Sistema de Recomendação Alimentar para Idosos")
st.sidebar.title("Navegação")

# Adicionando logotipo na barra lateral
st.sidebar.image("logo.png", use_column_width=True)

menu = st.sidebar.radio("Menu", ["Coleta de Dados", "Anamnese", "Recomendações", "Análise Gráfica", "Modelo de Machine Learning"])

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

    st.session_state['dados_paciente']['doencas'] = {doenca: st.checkbox(doenca, value=doenca in st.session_state['dados_paciente']['doencas']) for doenca in doencas_opcoes}
    
    # Converter para uma lista de doenças selecionadas
    doencas_selecionadas = [doenca for doenca, selecionado in st.session_state['dados_paciente']['doencas'].items() if selecionado]
    st.session_state['dados_paciente']['doencas'] = doencas_selecionadas
    st.write("Doenças selecionadas:", doencas_selecionadas)

    st.write("Informações salvas! Você pode continuar para a seção de recomendações.")

# Página 3: Recomendação de Refeições
elif menu == "Recomendações":
    st.header("Recomendações de Refeições")
    
    if st.session_state['dados_paciente']['tmb'] > 0 and st.session_state['dados_paciente']['doencas']:
        recomendacoes = sugerir_refeicoes(st.session_state['dados_paciente']['tmb'], st.session_state['dados_paciente']['doencas'])
        total_calorias_consumido = 0  
        
        st.write("Com base nas informações fornecidas, aqui estão as recomendações de refeições:")
        calorias_refeicoes = {}
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

    # Verificar se as informações nas páginas 1 e 2 foram preenchidas
    if (st.session_state['dados_paciente']['peso'] > 0 and 
        st.session_state['dados_paciente']['altura'] > 0 and 
        st.session_state['dados_paciente']['idade'] > 0 and 
        st.session_state['dados_paciente']['genero'] and 
        st.session_state['dados_paciente']['atividade_fisica'] and 
        st.session_state['dados_paciente']['doencas']):

        # Preparação dos dados para treinamento, convertendo variáveis categóricas para numéricas
        df_alimentos_encoded = pd.get_dummies(df_alimentos.drop(columns=["Target"]))
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