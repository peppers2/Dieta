import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
import alimentos

# load data
data = alimentos.todos_alimentos

# Choose columns to keep

columns = [
    'Numero', 'Alimento', 'Umidade', 'Calorias', 'Proteína', 'Lipídeos', 'Colesterol', 'Carboidrato', 'Cálcio', 'Sódio'
]

data = data[columns]
print(data.info())
print(data.describe())


# Generate and export the report
profile = ProfileReport(data, title='Relatório de Análise de Dados',
                        html={'style': {'full_width': True}}, sort=None)

profile.to_file("profile.html")
