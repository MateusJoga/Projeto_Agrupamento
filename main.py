#Base de dados: https://www.pns.icict.fiocruz.br/bases-de-dados/
#Dicionario da base: https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fwww.pns.icict.fiocruz.br%2Fwp-content%2Fuploads%2F2023%2F06%2Fdicionario_PNS_microdados_2019_23062023.xls&wdOrigin=BROWSELINK

import pandas as pd

nome_arquivo = '.\\dados\\pns2019.csv'

lista_variaveis = [
    'C006',      # Sexo
    'C008',      # Idade
    'V0001',     # UF
    'Q09201', 'Q09202', 'Q09203', 'Q09204', 'Q09205', 
    'Q09206', 'Q09207', 'Q09208', 'Q09209', # Perguntas PHQ-9
    'C009',      # Cor ou raça
    'VDD004A',   # Escolaridade
    'VDF004',    # Renda domiciliar per capita
    'E016',      # Situação de ocupação
    'C01001',    # Estado civil
    'P050',      # Autoavaliação da saúde
    'P00101',    # Diagnóstico de hipertensão
    'P00401',    # Diagnóstico de diabetes
    'P025',      # Diagnóstico de doença do coração
    'N001',      # Consumo de álcool
    'N007',      # Consumo de tabaco
    'O001'       # Prática de atividade física
]

df = pd.read_csv(nome_arquivo)

def listar_colunas(dataset):
    lista_de_colunas = dataset.columns.tolist()
    with open('lista_colunas.txt', 'w') as f:
        for nome_coluna in lista_de_colunas:
            f.write(f'{nome_coluna}\n')
    print('Concluído a lista!')
    return

'''
df_reduzido = df[lista_variaveis].copy()

print(df_reduzido.head())

nome = 'pns_reduzido.csv'

df_reduzido.to_csv(nome, index=False)
'''

listar_colunas(df)