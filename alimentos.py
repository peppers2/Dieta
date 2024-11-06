import pandas as pd
import numpy as np

pd.options.display.max_rows = None

tipos = [
    'Cereais e derivados',
    'Verduras, hortaliças e derivados',
    'Frutas e derivados',
    'Gorduras e óleos',
    'Pescados e frutos do mar',
    'Carnes e derivados',
    'Leite e derivados',
    'Bebidas (alcoólicas e não alcoólicas)',
    'Ovos e derivados',
    'Produtos açucarados',
    'Miscelâneas',
    'Outros alimentos industrializados',
    'Alimentos preparados',
    'Leguminosas e derivados',
    'Nozes e sementes'
]

columns = [
    'Numero', 'Nome', 'Umidade', 'Energia kcal', 'Proteína', 'Lipídeos', 'Colesterol', 'Carboidrato', 'Cálcio', 'Sódio'
]

taco = pd.read_excel(
    'data/Taco-4a-Edicao.xlsx', sheet_name='CMVCol taco3', skiprows=3, index_col=None, na_values=['NA'], usecols='A:D,F:H,I,L,R', names=columns)

cereais_derivados = pd.concat(
    [taco.iloc[:31], taco.iloc[35:66]], ignore_index=True)
cereais_derivados['Tipo'] = 'Cereais e derivados'
print(cereais_derivados)


verduras_hortalicas_derivados = pd.concat(
    [
        taco.iloc[71:101],
        taco.iloc[104:136],
        taco.iloc[139:171],
        taco.iloc[174:179],
    ], ignore_index=True)
verduras_hortalicas_derivados['Tipo'] = 'Verduras, hortaliças e derivados'
print(verduras_hortalicas_derivados)


frutas_derivados = pd.concat(
    [
        taco.iloc[181:206],
        taco.iloc[209:241],
        taco.iloc[244:276],
        taco.iloc[279:286],
    ], ignore_index=True)
frutas_derivados['Tipo'] = 'Frutas e derivados'
print(frutas_derivados)


gorduras_oleos = pd.concat(
    [
        taco.iloc[288:302],
    ], ignore_index=True)
gorduras_oleos['Tipo'] = 'Gorduras e óleos'
print(gorduras_oleos)


pescados_frutos_do_mar = pd.concat(
    [
        taco.iloc[305:311],
        taco.iloc[314:346],
        taco.iloc[349:361],
    ], ignore_index=True)
pescados_frutos_do_mar['Tipo'] = 'Pescados e frutos do mar'
print(pescados_frutos_do_mar)


carnes_derivados = pd.concat(
    [
        taco.iloc[363:381],
        taco.iloc[384:416],
        taco.iloc[419:451],
        taco.iloc[454:486],
        taco.iloc[489:498],
    ], ignore_index=True)
carnes_derivados['Tipo'] = 'Carnes e derivados'
print(carnes_derivados)


leite_derivados = pd.concat(
    [
        taco.iloc[500:520],
        taco.iloc[523:527],
    ], ignore_index=True)
leite_derivados['Tipo'] = 'Leite e derivados'
print(leite_derivados)


bebidas = pd.concat(
    [
        taco.iloc[529:543]
    ], ignore_index=True)
bebidas['Tipo'] = 'Bebidas (alcoólicas e não alcoólicas)'
print(bebidas)


ovos_derivados = pd.concat(
    [
        taco.iloc[545:552],
    ], ignore_index=True)
ovos_derivados['Tipo'] = 'Ovos e derivados'
print(ovos_derivados)


produtos_acucarados = pd.concat(
    [
        taco.iloc[553:555],
        taco.iloc[558:576]
    ], ignore_index=True)
produtos_acucarados['Tipo'] = 'Produtos açucarados'
print(produtos_acucarados)


miscelaneas = pd.concat(
    [
        taco.iloc[578:587]
    ], ignore_index=True)
miscelaneas['Tipo'] = 'Miscelâneas'
print(miscelaneas)


outros_alimentos_industrializados = pd.concat(
    [
        taco.iloc[[589]],
        taco.iloc[593:597],
    ], ignore_index=True)
outros_alimentos_industrializados['Tipo'] = 'Outros alimentos industrializados'
print(outros_alimentos_industrializados)


alimentos_preparados = pd.concat(
    [
        taco.iloc[599:625],
        taco.iloc[628:635],
    ], ignore_index=True)
alimentos_preparados['Tipo'] = 'Alimentos preparados'
print(alimentos_preparados)


leguminosas_derivados = pd.concat(
    [
        taco.iloc[637:658],
        taco.iloc[661:670],
    ], ignore_index=True)
leguminosas_derivados['Tipo'] = 'Leguminosas e derivados'
print(leguminosas_derivados)


nozes_sementes = pd.concat(
    [
        taco.iloc[672:683]
    ], ignore_index=True)
nozes_sementes['Tipo'] = 'Nozes e sementes'
print(nozes_sementes)


todos_alimentos = pd.concat([cereais_derivados, verduras_hortalicas_derivados, frutas_derivados, gorduras_oleos, pescados_frutos_do_mar, miscelaneas,
                             carnes_derivados, leite_derivados, bebidas, ovos_derivados, produtos_acucarados, outros_alimentos_industrializados, alimentos_preparados, leguminosas_derivados, nozes_sementes], ignore_index=True)

todos_alimentos.fillna(0, inplace=True)
todos_alimentos = todos_alimentos.replace('Tr', 0)

# delete lines with values equal to *
todos_alimentos = todos_alimentos.replace('*', np.nan)
todos_alimentos = todos_alimentos.dropna()

int_cols = ['Energia kcal']

todos_alimentos[int_cols] = todos_alimentos[int_cols].apply(
    lambda x: x.astype(int))

print('Todos alimentos')
print(todos_alimentos)

print(todos_alimentos.info())


todos_alimentos.to_csv('data/todos_alimentos.csv', index=False)

doencas = ["Diabetes Tipo 1", "Diabetes Tipo 2", "Hipertensão", "Obesidade", "Diabetes Tipo 1", "Diabetes Tipo 2", "Hipertensão", "Obesidade",
           "Insuficiência Renal", "Colesterol Alto", "Doenças Cardíacas",
           "Osteoporose", "Doença Celíaca", "Nenhuma"]


# Adiciona uma coluna 'Doença' com doenças aleatórias
todos_alimentos['Doencas_Restritivas'] = np.random.choice(
    doencas, size=len(todos_alimentos))

refeicao_indicada = ["Café da Manhã", "Almoço", "Lanche da Tarde", "Jantar"]

# Adiciona uma coluna 'Refeicao_Indicada' com tipos de refeição aleatórios
todos_alimentos['Refeicao_Indicada'] = np.random.choice(
    refeicao_indicada, size=len(todos_alimentos))


# Renomeia nome e energia para manter compatibilidade com o modelo
todos_alimentos.rename(columns={'Nome': 'Alimento',
                                'Energia kcal': 'Calorias'}, inplace=True)

# Salva o dataframe atualizado com tipos de refeição
todos_alimentos.to_csv(
    'data/alimentos.csv', index=False)
