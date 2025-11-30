#Base de dados: https://www.pns.icict.fiocruz.br/bases-de-dados/
#Dicionario da base: https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fwww.pns.icict.fiocruz.br%2Fwp-content%2Fuploads%2F2023%2F06%2Fdicionario_PNS_microdados_2019_23062023.xls&wdOrigin=BROWSELINK
#Dataset do basedosdados: https://basedosdados.org/dataset/86bac6cc-575f-4289-a857-13f3f52c9a1d?table=f8b6030c-3fb1-4f64-81ef-3c5c3c888b6c

import pandas as pd
import numpy as np

nome_arquivo = '.\\dados\\pns_teste_soma.csv'

lista_variaveis = [
    # ----------------------------------------
    # I. Fatores Geográficos e Demográficos
    # ----------------------------------------
    'V0001',     # Unidade da Federação (UF) [1, 2]
    'C006',      # Sexo [3]
    'C008',      # Idade do morador na data de referência (em anos) [4]
    'C009',      # Cor ou raça (Branca, Preta, Amarela, Parda, Indígena) [4]
    'C01001',    # Cônjuge ou companheiro(a) mora em nesse domicílio [4]
    'D00901',    # Curso mais elevado que ___ frequentou (Escolaridade) [5, 6]
    
    # ----------------------------------------
    # II. Fatores Socioeconômicos
    # ----------------------------------------
    'E01401',    # Posição na ocupação (Empregado, Conta própria, Empregador, etc.) [7, 8]
    'E01602',    # Rendimento bruto mensal ou retirada que ___ fazia normalmente nesse trabalho (Valor em R$) [9, 10]
    'A01901',    # Algum morador tem acesso à Internet no domicílio [11, 12]
    
    # ----------------------------------------
    # III. Fatores de Saúde e Comportamento (Estilo de Vida e Comorbidades)
    # ----------------------------------------
    'J00101',    # Avaliação do estado de saúde, considerando bem-estar físico e mental [13]
    'Q00201',    # Diagnóstico de hipertensão arterial (pressão alta) [14]
    'Q03001',    # Diagnóstico de diabetes [15]
    'Q06306',    # Diagnóstico de doença do coração (infarto, angina, insuficiência cardíaca ou outra) [16]
    'P027',      # Frequência de consumo de bebida alcoólica [17]
    'P034',      # Nos últimos três meses, praticou algum tipo de exercício físico ou esporte? [18]
    'P050',      # Atualmente, o(a) Sr(a) fuma algum produto do tabaco? [19]
    
    # ----------------------------------------
    # IV. Variáveis de Depressão (Diagnóstico e Sintomas PHQ-9)
    # ----------------------------------------
    'N010',      # Problemas no sono nas duas últimas semanas [21]
    'N011',      # Sentiu-se cansado(a) ou sem energia nas duas últimas semanas [21]
    'N012',      # Pouco interesse ou não sentir prazer em fazer as coisas nas duas últimas semanas [21]
    'N013',      # Problemas para se concentrar nas suas atividades habituais nas duas últimas semanas [22]
    'N014',      # Problemas na alimentação (falta de apetite ou comer muito mais) nas duas últimas semanas [23]
    'N015',      # Lentidão ou agitação/inquietude nas duas últimas semanas [24]
    'N016',      # Sentiu-se deprimido(a), “pra baixo” ou sem perspectiva nas duas últimas semanas [24]
    'N017',      # Sentiu-se mal consigo mesmo, achando-se um fracasso nas duas últimas semanas [24]
    'N018'       # Pensou em se ferir ou achou que seria melhor estar morto nas duas últimas semanas [25]
]

df = pd.read_csv(nome_arquivo)

def listar_colunas(dataset):
    lista_de_colunas = dataset.columns.tolist()
    with open('lista_colunas.txt', 'w') as f:
        for nome_coluna in lista_de_colunas:
            f.write(f'{nome_coluna}\n')
    print('Concluído a lista!')
    return

def score_PHQ9(dataset,colunas):
    #Colunas usadas:
    #colunas_chave = ['N010', 'N011', 'N012', 'N013', 'N014', 'N015', 'N016', 'N017', 'N018']

    df_recodificado = dataset.copy()

    #Transformamos a variável em numérica para retirarmos 1 ponto do atributo para entrar no contexto de pontuação de 0 a 3
    for coluna in colunas:
        df_recodificado[coluna] = pd.to_numeric(df_recodificado[coluna], errors='coerce')
        df_recodificado[coluna] = df_recodificado[coluna].apply(lambda x:x -1 if pd.notna(x) and x in [1,2,3,4] else np.nan)

    #Criamos a variável de Score para o PHQ
    df_recodificado['SCR_PHQ9'] = df_recodificado[colunas].sum(axis=1)

    return df_recodificado.to_csv('pns_teste_soma.csv', index=False)

def categ_PHQ9(dataset):
    bins = [-1, 4, 9, 14, 19, 27] 

    # Rótulos das Categorias:
    labels = [
        'Mínima (0-4)', 
        'Leve (5-9)', 
        'Moderada (10-14)', 
        'Moderadamente Grave (15-19)', 
        'Grave (20-27)'
    ]

    # 2. Crie a nova variável Categórica (CAT_PHQ9)
    # Usamos pd.cut() para criar a variável categórica com base nos limites definidos.
    # O parâmetro 'right=True' significa que os limites superiores do intervalo são INCLUÍDOS (ex: 4, 9, 14, etc.).
    # Isso corresponde exatamente às faixas que você especificou.
    dataset['CAT_PHQ9'] = pd.cut(
        dataset['SCR_PHQ9'], 
        bins=bins, 
        labels=labels, 
        right=True,
        ordered=True  # Indica que a variável categórica é ordinal (tem ordem)
    )

    return dataset.to_csv('pns_teste_categ.csv', index=False)

categ_PHQ9(df)

'''
df_selecionado = df[lista_variaveis].copy()

colunas_chave = [
    'N010', 'N011', 'N012', 'N013', 'N014', 'N015', 'N016', 'N017', 'N018',
     'C006', 'C008', 'C009', 'C01001', 'D00901', 'E01401', 'E01602', 'A01901',
      'Q00201', 'J00101', 'Q03001', 'Q06306', 'P027', 'P034', 'P050' 
]
# 3. Remova todas as linhas que contêm NaN em QUALQUER das colunas-chave
df_limpo = df_selecionado.dropna(subset=colunas_chave)

df_limpo.to_csv('pns_teste_soma.csv', index=False)
'''