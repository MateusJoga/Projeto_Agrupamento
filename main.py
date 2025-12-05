#Base de dados: https://www.pns.icict.fiocruz.br/bases-de-dados/
#Dicionario da base: https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fwww.pns.icict.fiocruz.br%2Fwp-content%2Fuploads%2F2023%2F06%2Fdicionario_PNS_microdados_2019_23062023.xls&wdOrigin=BROWSELINK
#Dataset do basedosdados: https://basedosdados.org/dataset/86bac6cc-575f-4289-a857-13f3f52c9a1d?table=f8b6030c-3fb1-4f64-81ef-3c5c3c888b6c

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

nome_arquivo = '.\\dados\\pns_categ.csv'

lista_variaveis = [    
    
    # ========================================
    # II. FATORES DEMOGRÁFICOS E GEOGRÁFICOS
    # ========================================
    'V0001',     # Unidade da Federação (UF) [6]
    'V0031',     # Tipo de área (Urbano/Rural) [7]
    'C006',      # Sexo [8]
    'C008',      # Idade do morador na data de referência (em anos) [9]
    'C009',      # Cor ou raça (Branca, Preta, Parda, etc.) [9]
    'C011',      # Estado civil (Casado(a), Divorciado(a), Solteiro(a), etc.) [10]
    
    # ========================================
    # III. FATORES SOCIOECONÔMICOS (Priorizando Variáveis Derivadas)
    # ========================================
    'VDD004A',   # Nível de instrução mais elevado alcançado (Escolaridade padronizada) [11]
    'VDF004',    # Faixa de rendimento domiciliar per capita (Faixas em salários mínimos) [13]
    'E01602',    # Rendimento bruto mensal do trabalho principal (Valor em R$) [14]
    'A01901',    # Acesso à Internet no domicílio [15]
    
    # ========================================
    # IV. FATORES SAÚDE / COMPORTAMENTO / COMORBIDADES
    # ========================================
    'J00101',    # Avaliação do estado de saúde (físico e mental) [16]
    'Q00201',    # Diagnóstico de Hipertensão arterial [19]
    'Q03001',    # Diagnóstico de Diabetes [20]
    'Q06306',    # Diagnóstico de Doença do coração [21]
    'Q084',      # Problema crônico de coluna (Dor crônica, lombalgia, etc.) [22]
    'Q132',      # Nas últimas duas semanas, fez uso de algum medicamento para dormir [23]
    'P027',      # Frequência de consumo de bebida alcoólica [24]
    'P034',      # Nos últimos três meses, praticou exercício físico ou esporte [25]
    'P050',      # Atualmente, fuma algum produto do tabaco [26]
    'M01401',    # Com quantos familiares ou parentes próximos pode contar (Apoio Social Familiar) [27]

    # I. VARIÁVEIS DE SINTOMAS PHQ-9 (TREINAMENTO CENTRAL)
    # Estas variáveis medem a frequência dos 9 sintomas nas últimas 2 semanas.
    # ========================================
    'N010',      # Problemas no sono (dificuldade para adormecer, etc.) [1]
    'N011',      # Sentiu-se cansado(a) ou sem energia [1]
    'N012',      # Pouco interesse ou não sentir prazer em fazer as coisas [1]
    'N013',      # Problemas para se concentrar [2]
    'N014',      # Problemas na alimentação (falta de apetite ou comer muito mais) [3]
    'N015',      # Lentidão ou agitação/inquietude [4]
    'N016',      # Sentiu-se deprimido(a), “pra baixo” ou sem perspectiva [4]
    'N017',      # Sentiu-se mal consigo mesmo, achando-se um fracasso [4]
    'N018',      # Pensou em se ferir ou achou que seria melhor estar morto [5]

    # V. VARIÁVEIS DE COMPARAÇÃO
    'V03502',    # Ato sexual forçado gerou consequências psicológicas (inclui depressão, medo, tristeza) [28]
    'Q092',      # Diagnóstico anterior de Depressão (por médico ou profissional de saúde mental) [17]
    'Q109',      # Grau em que a depressão limita as atividades habituais [18]
]


def listar_colunas(dataset):
    lista_de_colunas = dataset.columns.tolist()
    with open('lista_colunas.txt', 'w') as f:
        for nome_coluna in lista_de_colunas:
            f.write(f'{nome_coluna}\n')
    print('Concluído a lista!')
    return

def score_PHQ9(dataset,colunas):    

    df_recodificado = dataset.copy()

    #Transformamos a variável em numérica para retirarmos 1 ponto do atributo para entrar no contexto de pontuação de 0 a 3
    for coluna in colunas:
        df_recodificado[coluna] = pd.to_numeric(df_recodificado[coluna], errors='coerce')
        df_recodificado[coluna] = df_recodificado[coluna].apply(lambda x:x -1 if pd.notna(x) and x in [1,2,3,4] else np.nan)

    #Criamos a variável de Score para o PHQ
    df_recodificado['SCR_PHQ9'] = df_recodificado[colunas].sum(axis=1)
    df_recodificado.to_csv('pns_soma.csv', index=False)

    return df_recodificado

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
    dataset['CAT_PHQ9'] = pd.cut(
        dataset['SCR_PHQ9'], 
        bins=bins, 
        labels=labels, 
        right=True,
        ordered=True  # Indica que a variável categórica é ordinal (tem ordem)
    )

    dataset.to_csv('pns_categ.csv', index=False)

    return dataset

def novo_df_registro_limpo(df):
    lista = [
    # ========================================
    # I. VARIÁVEIS DE SINTOMAS PHQ-9 (TREINAMENTO CENTRAL)
    # Estas variáveis medem a frequência dos 9 sintomas nas últimas 2 semanas.
    # ========================================
    'N010',      # Problemas no sono (dificuldade para adormecer, etc.) [1]
    'N011',      # Sentiu-se cansado(a) ou sem energia [1]
    'N012',      # Pouco interesse ou não sentir prazer em fazer as coisas [1]
    'N013',      # Problemas para se concentrar [2]
    'N014',      # Problemas na alimentação (falta de apetite ou comer muito mais) [3]
    'N015',      # Lentidão ou agitação/inquietude [4]
    'N016',      # Sentiu-se deprimido(a), “pra baixo” ou sem perspectiva [4]
    'N017',      # Sentiu-se mal consigo mesmo, achando-se um fracasso [4]
    'N018',      # Pensou em se ferir ou achou que seria melhor estar morto [5]
    
    # ========================================
    # II. FATORES DEMOGRÁFICOS E GEOGRÁFICOS
    # ========================================
    'V0001',     # Unidade da Federação (UF) [6]
    'V0031',     # Tipo de área (Urbano/Rural) [7]
    'C006',      # Sexo [8]
    'C008',      # Idade do morador na data de referência (em anos) [9]
    'C009',      # Cor ou raça (Branca, Preta, Parda, etc.) [9]
    'C011',      # Estado civil (Casado(a), Divorciado(a), Solteiro(a), etc.) [10]
    
    # ========================================
    # III. FATORES SOCIOECONÔMICOS (Priorizando Variáveis Derivadas)
    # ========================================
    'VDD004A',   # Nível de instrução mais elevado alcançado (Escolaridade padronizada) [11]
    'VDF004',    # Faixa de rendimento domiciliar per capita (Faixas em salários mínimos) [13]
    'E01602',    # Rendimento bruto mensal do trabalho principal (Valor em R$) [14]
    'A01901',    # Acesso à Internet no domicílio [15]
    
    # ========================================
    # IV. FATORES SAÚDE / COMPORTAMENTO / COMORBIDADES
    # ========================================
    'J00101',    # Avaliação do estado de saúde (físico e mental) [16]
    'Q00201',    # Diagnóstico de Hipertensão arterial [19]
    'Q03001',    # Diagnóstico de Diabetes [20]
    'Q06306',    # Diagnóstico de Doença do coração [21]
    'Q084',      # Problema crônico de coluna (Dor crônica, lombalgia, etc.) [22]
    'Q132',      # Nas últimas duas semanas, fez uso de algum medicamento para dormir [23]
    'P027',      # Frequência de consumo de bebida alcoólica [24]
    'P034',      # Nos últimos três meses, praticou exercício físico ou esporte [25]
    'P050',      # Atualmente, fuma algum produto do tabaco [26]
    'M01401',    # Com quantos familiares ou parentes próximos pode contar (Apoio Social Familiar) [27]
]
    df_selecionado = df[lista_variaveis].copy()
    df_limpo = df_selecionado.dropna(subset=lista)
    return df_limpo

def OneHotEnconding(df, cols):
    print(f"Aplicando One-Hot Encoding em {len(cols)} colunas: {cols}")
    df_dummies = pd.get_dummies(df, columns=cols, prefix=cols, dummy_na=False)
    colunas_booleanas = df_dummies.select_dtypes(include=['bool']).columns
    df_dummies[colunas_booleanas] = df_dummies[colunas_booleanas].astype(int)
    print(f"OHE concluído. O DataFrame agora tem {df_dummies.shape[1]} colunas.")
    return df_dummies

def pre_process_ordinais(df, cols):
    
    df_temp = df.copy()
    
    print(f"Iniciando pré-processamento das {len(cols)} variáveis ordinais...")

    for coluna in cols:
        if coluna in df_temp.columns:
            
            df_temp[coluna] = pd.to_numeric(df_temp[coluna], errors='coerce')
            
            if coluna == 'J00101': 
                print(f"   -> Aplicando INVERSÃO de escala em {coluna}")
                # A inversão só deve ser aplicada a valores válidos (entre 1 e 5), evitando NaN e outros códigos
                df_temp.loc[df_temp[coluna].between(1, 5, inclusive='both'), coluna] = 6 - df_temp[coluna]

            if coluna == 'P050': 
                print(f"   -> Aplicando INVERSÃO de escala em {coluna}")
                # A inversão só deve ser aplicada a valores válidos (entre 1 e 3), evitando NaN e outros códigos
                df_temp.loc[df_temp[coluna].between(1, 3, inclusive='both'), coluna] = 4 - df_temp[coluna]

        else:
            print(f"Aviso: Coluna {coluna} não encontrada no DataFrame.")
    
    print("Pré-processamento das variáveis ordinais concluído com sucesso!")
    return df_temp

def preprocess_scaling(df, cols):
    
    df_temp = df.copy()
    colunas_presentes = [col for col in cols if col in df_temp.columns]
    
    if not colunas_presentes:
        print("Aviso: Nenhuma coluna numérica encontrada para aplicar o StandardScaler.")
        return df_temp

    print(f"Aplicando StandardScaler em {len(colunas_presentes)} variáveis numéricas (Contínuas e Ordinais)...")
    scaler = StandardScaler()
    df_temp[colunas_presentes] = scaler.fit_transform(df_temp[colunas_presentes])

    print("Padronização concluída. Todas as variáveis numéricas agora têm média 0 e desvio-padrão 1.")
    return df_temp

def realizar_clusterizacao_e_plotagem_final(df,cols, n_comp, n_clusters, random_state: int = 42):
    
    colunas_para_excluir_presentes = [col for col in cols if col in df.columns]
    X_final = df.drop(columns=colunas_para_excluir_presentes, errors='ignore')
    
    # 3. Treinamento do Modelo K-Means
    pca = PCA(n_components=n_comp, random_state=42)
    X_red = pca.fit_transform(X_final)
    
    kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=random_state)
    kmeans_model.fit(X_red)
    df['Cluster_Label'] = kmeans_model.labels_
    contagem_clusters = df['Cluster_Label'].value_counts().sort_index()
    print(contagem_clusters)
    # 4. ANÁLISE DE PERFIL - OUTCOME CATEGÓRICO (CAT_PHQ9)
    if 'CAT_PHQ9' in df.columns:
        print("\n[Tabela 1] Distribuição do Outcome (CAT_PHQ9) por Cluster (Proporção %):")
        crosstab_phq9 = pd.crosstab(df['Cluster_Label'], df['CAT_PHQ9'], normalize='index') * 100
        perfil_categorico = crosstab_phq9.round(2)
        print(perfil_categorico)

        # 5. PLOTAGEM DO PRINCIPAL INSIGHT: Distribuição de Depressão por Cluster
        perfil_categorico.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title('Distribuição da Gravidade da Depressão (CAT_PHQ9) por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Proporção (%)')
        plt.xticks(rotation=0)
        plt.legend(title='Gravidade PHQ-9', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    else:
        print("Aviso: CAT_PHQ9 não encontrada para análise categórica.")
        
    # 6. ANÁLISE DE PERFIL - FEATURES NUMÉRICAS/ORDINAIS (Para a caracterização)
    COLUNAS_CODIGO_ANALISE = list(COLUNAS_ROTULADA.keys())
    df_analise = df[['Cluster_Label'] + [col for col in COLUNAS_CODIGO_ANALISE if col in df.columns]]
    perfil_numerico = df_analise.groupby('Cluster_Label').mean().round(2)
    
    perfil_numerico = perfil_numerico.rename(columns=COLUNAS_ROTULADA)
    perfil_numerico_transposto = perfil_numerico.T

    print("\n[Tabela 2] Perfil Numérico/Ordinal (Média dos Valores Scaled/Originais por Cluster):")
    print(perfil_numerico_transposto)
    
    perfil_categorico_df = pd.DataFrame()
    colunas_ohe_mapeadas = {} # Dicionário para mapear as colunas OHE de volta para o rótulo original

    print("\n[Tabela 3] Perfil Categórico OHE (Proporção % de Categorias por Cluster):")
    for codigo_original, rotulo_original in COLUNAS_ROT_CAT.items():
        # Encontra todas as colunas que começam com o código original (são as colunas OHE)
        colunas_ohe = [col for col in df.columns if col.startswith(codigo_original)]
        
        if colunas_ohe:
            df_ohe_cluster = df.groupby('Cluster_Label')[colunas_ohe].mean() * 100
            
            # Renomeia as colunas OHE para o formato 'Rótulo Original: Categoria'
            # Ex: C006_1 -> Sexo: Masculino (se 1 é Masculino)
            mapeamento_colunas = {
                col: f"{rotulo_original}: {col.replace(codigo_original, '').lstrip('_')}" 
                for col in colunas_ohe
            }
            df_ohe_cluster = df_ohe_cluster.rename(columns=mapeamento_colunas).round(1)
            
            # Adiciona ao DataFrame de perfil, garantindo a concatenação correta
            perfil_categorico_df = pd.concat([perfil_categorico_df, df_ohe_cluster.T])
            colunas_ohe_mapeadas.update(mapeamento_colunas) # Guarda para o resumo final

    if not perfil_categorico_df.empty:
        print(perfil_categorico_df)
    else:
        print("Nenhuma coluna OHE encontrada com os prefixos especificados em COLUNAS_ROT_CAT.")

    #======================
    relevancia_cluster = {}
    
    # 5a. Calcula a Média Geral para as colunas OHE (proporção na população total)
    medias_ohe_populacao = df[list(colunas_ohe_mapeadas.keys())].mean() * 100
    medias_ohe_populacao.index = [colunas_ohe_mapeadas.get(col, col) for col in medias_ohe_populacao.index]
    
    for cluster_label in range(n_clusters):
        relevancia = {}

        # 5b. Relevância Numérica/Ordinal (Magnitude do Desvio-Padrão - |Média| > 0.5)
        medias_cluster_num = perfil_numerico_transposto[cluster_label]
        # Filtrar por desvio maior que 0.5
        relevancia_numerica = medias_cluster_num[medias_cluster_num.abs() > 0.5]
        
        for feature, valor in relevancia_numerica.items():
            relevancia[feature] = (abs(valor), f"{valor:.2f} σ") # (Métrica de ordenação, Rótulo)
            
        # 5c. Relevância Categórica OHE (Magnitude da Diferença em Relação à Média Geral)
        if not perfil_categorico_df.empty:
            # Seleciona o perfil do cluster para as categorias OHE
            perfil_cluster_ohe = perfil_categorico_df[cluster_label] 
            
            for feature_label in perfil_cluster_ohe.index:
                valor_cluster = perfil_cluster_ohe.loc[feature_label]
                valor_populacao = medias_ohe_populacao.loc[feature_label]
                
                # Relevância é a diferença absoluta da proporção da população
                diferenca = valor_cluster - valor_populacao
                
                # Se a diferença absoluta for maior que 10% (um bom limiar)
                if abs(diferenca) > 10:
                    # Rótulo: Exibe a proporção do cluster e a diferença em %
                    rotulo = f"{valor_cluster:.1f}% ({'+' if diferenca > 0 else ''}{diferenca:.1f}p.p.)"
                    relevancia[feature_label] = (abs(diferenca), rotulo)

        # 5d. Ordenar todas as variáveis (Numéricas e Categóricas)
        relevancia_ordenada = {
            k: v[1] for k, v in sorted(relevancia.items(), key=lambda item: item[1][0], reverse=True)
        }
        relevancia_cluster[f'Cluster {cluster_label}'] = relevancia_ordenada
        
    print("\n[Etapa 8] RESUMO: Variáveis de Maior Relevância por Cluster (Ordem Decrescente):")
    for cluster, variaveis in relevancia_cluster.items():
        print(f"\n--- {cluster} ---")
        if variaveis:
            for k, v in variaveis.items():
                print(f"- {k}: {v}")
        else:
            print("- Nenhuma variável significativa (desvio > 0.5σ ou diferença > 10p.p.)")

    return df, perfil_numerico, perfil_categorico

def realizar_clusterizacao_e_plotagem_sempca(df,cols, n_clusters, random_state: int = 42):
    
    colunas_para_excluir_presentes = [col for col in cols if col in df.columns]
    X_final = df.drop(columns=colunas_para_excluir_presentes, errors='ignore')

    kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=random_state)
    kmeans_model.fit(X_final)
    df['Cluster_Label'] = kmeans_model.labels_
    contagem_clusters = df['Cluster_Label'].value_counts().sort_index()
    print(contagem_clusters)
    # 4. ANÁLISE DE PERFIL - OUTCOME CATEGÓRICO (CAT_PHQ9)
    if 'CAT_PHQ9' in df.columns:
        print("\n[Tabela 1] Distribuição do Outcome (CAT_PHQ9) por Cluster (Proporção %):")
        crosstab_phq9 = pd.crosstab(df['Cluster_Label'], df['CAT_PHQ9'], normalize='index') * 100
        perfil_categorico = crosstab_phq9.round(2)
        print(perfil_categorico)
        pd.set_option('display.max_columns', None)
        # 5. PLOTAGEM DO PRINCIPAL INSIGHT: Distribuição de Depressão por Cluster
        perfil_categorico.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title('Distribuição da Gravidade da Depressão (CAT_PHQ9) por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Proporção (%)')
        plt.xticks(rotation=0)
        plt.legend(title='Gravidade PHQ-9', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    else:
        print("Aviso: CAT_PHQ9 não encontrada para análise categórica.")
        
    # 6. ANÁLISE DE PERFIL - FEATURES NUMÉRICAS/ORDINAIS (Para a caracterização)
    COLUNAS_CODIGO_ANALISE = list(COLUNAS_ROTULADA.keys())
    df_analise = df[['Cluster_Label'] + [col for col in COLUNAS_CODIGO_ANALISE if col in df.columns]]
    perfil_numerico = df_analise.groupby('Cluster_Label').mean().round(2)
    
    perfil_numerico = perfil_numerico.rename(columns=COLUNAS_ROTULADA)
    perfil_numerico_transposto = perfil_numerico.T

    print("\n[Tabela 2] Perfil Numérico/Ordinal (Média dos Valores Scaled/Originais por Cluster):")
    print(perfil_numerico_transposto)

    perfil_categorico_df = pd.DataFrame()
    colunas_ohe_mapeadas = {} # Dicionário para mapear as colunas OHE de volta para o rótulo original

    print("\n[Tabela 3] Perfil Categórico OHE (Proporção % de Categorias por Cluster):")
    for codigo_original, rotulo_original in COLUNAS_ROT_CAT.items():
        # Encontra todas as colunas que começam com o código original (são as colunas OHE)
        colunas_ohe = [col for col in df.columns if col.startswith(codigo_original)]
        
        if colunas_ohe:
            df_ohe_cluster = df.groupby('Cluster_Label')[colunas_ohe].mean() * 100
            
            # Renomeia as colunas OHE para o formato 'Rótulo Original: Categoria'
            # Ex: C006_1 -> Sexo: Masculino (se 1 é Masculino)
            mapeamento_colunas = {
                col: f"{rotulo_original}: {col.replace(codigo_original, '').lstrip('_')}" 
                for col in colunas_ohe
            }
            df_ohe_cluster = df_ohe_cluster.rename(columns=mapeamento_colunas).round(1)
            
            # Adiciona ao DataFrame de perfil, garantindo a concatenação correta
            perfil_categorico_df = pd.concat([perfil_categorico_df, df_ohe_cluster.T])
            colunas_ohe_mapeadas.update(mapeamento_colunas) # Guarda para o resumo final

    if not perfil_categorico_df.empty:
        print(perfil_categorico_df)
    else:
        print("Nenhuma coluna OHE encontrada com os prefixos especificados em COLUNAS_ROT_CAT.")

    #======================
    relevancia_cluster = {}
    
    # 5a. Calcula a Média Geral para as colunas OHE (proporção na população total)
    medias_ohe_populacao = df[list(colunas_ohe_mapeadas.keys())].mean() * 100
    medias_ohe_populacao.index = [colunas_ohe_mapeadas.get(col, col) for col in medias_ohe_populacao.index]
    
    for cluster_label in range(n_clusters):
        relevancia = {}

        # 5b. Relevância Numérica/Ordinal (Magnitude do Desvio-Padrão - |Média| > 0.5)
        medias_cluster_num = perfil_numerico_transposto[cluster_label]
        # Filtrar por desvio maior que 0.5
        relevancia_numerica = medias_cluster_num[medias_cluster_num.abs() > 0.5]
        
        for feature, valor in relevancia_numerica.items():
            relevancia[feature] = (abs(valor), f"{valor:.2f} σ") # (Métrica de ordenação, Rótulo)
            
        # 5c. Relevância Categórica OHE (Magnitude da Diferença em Relação à Média Geral)
        if not perfil_categorico_df.empty:
            # Seleciona o perfil do cluster para as categorias OHE
            perfil_cluster_ohe = perfil_categorico_df[cluster_label] 
            
            for feature_label in perfil_cluster_ohe.index:
                valor_cluster = perfil_cluster_ohe.loc[feature_label]
                valor_populacao = medias_ohe_populacao.loc[feature_label]
                
                # Relevância é a diferença absoluta da proporção da população
                diferenca = valor_cluster - valor_populacao
                
                # Se a diferença absoluta for maior que 10% (um bom limiar)
                if abs(diferenca) > 10:
                    # Rótulo: Exibe a proporção do cluster e a diferença em %
                    rotulo = f"{valor_cluster:.1f}% ({'+' if diferenca > 0 else ''}{diferenca:.1f}p.p.)"
                    relevancia[feature_label] = (abs(diferenca), rotulo)

        # 5d. Ordenar todas as variáveis (Numéricas e Categóricas)
        relevancia_ordenada = {
            k: v[1] for k, v in sorted(relevancia.items(), key=lambda item: item[1][0], reverse=True)
        }
        relevancia_cluster[f'Cluster {cluster_label}'] = relevancia_ordenada
        
    print("\n[Etapa 8] RESUMO: Variáveis de Maior Relevância por Cluster (Ordem Decrescente):")
    for cluster, variaveis in relevancia_cluster.items():
        print(f"\n--- {cluster} ---")
        if variaveis:
            for k, v in variaveis.items():
                print(f"- {k}: {v}")
        else:
            print("- Nenhuma variável significativa (desvio > 0.5σ ou diferença > 10p.p.)")
    
    return df, perfil_numerico, perfil_categorico

def realizar_clusterizacao_KMEDOID(df, cols, n_clusters, random_state=42):
    colunas_para_excluir_presentes = [col for col in cols if col in df.columns]
    X_final = df.drop(columns=colunas_para_excluir_presentes, errors='ignore')

    pca = PCA(n_components=8, random_state=42)
    X_red = pca.fit_transform(X_final)


    kmedoids_model = KMedoids(
    n_clusters=n_clusters, 
    method='pam', 
    max_iter=300, 
    random_state=random_state,
    # distance='euclidean' é o padrão, mas você pode testar 'manhattan' se desejar
    )
    kmedoids_model.fit(X_red)
    df['Cluster_Label'] = kmedoids_model.labels_
    if 'CAT_PHQ9' in df.columns:
        print("\n[Tabela 1] Distribuição do Outcome (CAT_PHQ9) por Cluster (Proporção %):")
        crosstab_phq9 = pd.crosstab(df['Cluster_Label'], df['CAT_PHQ9'], normalize='index') * 100
        perfil_categorico = crosstab_phq9.round(2)
        print(perfil_categorico)

        # 5. PLOTAGEM DO PRINCIPAL INSIGHT: Distribuição de Depressão por Cluster
        perfil_categorico.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title('Distribuição da Gravidade da Depressão (CAT_PHQ9) por Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Proporção (%)')
        plt.xticks(rotation=0)
        plt.legend(title='Gravidade PHQ-9', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    else:
        print("Aviso: CAT_PHQ9 não encontrada para análise categórica.")
        
    # 6. ANÁLISE DE PERFIL - FEATURES NUMÉRICAS/ORDINAIS (Para a caracterização)
    COLUNAS_CODIGO_ANALISE = list(COLUNAS_ROTULADA.keys())
    df_analise = df[['Cluster_Label'] + [col for col in COLUNAS_CODIGO_ANALISE if col in df.columns]]
    perfil_numerico = df_analise.groupby('Cluster_Label').mean().round(2)
    
    perfil_numerico = perfil_numerico.rename(columns=COLUNAS_ROTULADA)
    perfil_numerico_transposto = perfil_numerico.T

    print("\n[Tabela 2] Perfil Numérico/Ordinal (Média dos Valores Scaled/Originais por Cluster):")
    print(perfil_numerico_transposto)

    perfil_categorico_df = pd.DataFrame()
    
    for col in COLUNAS_ROT_CAT:
        if col in df.columns:
            # Calcula a proporção de cada categoria dentro de cada cluster
            crosstab_cat = pd.crosstab(df['Cluster_Label'], df[col], normalize='index') * 100
            
            # Formata para incluir a variável no índice
            crosstab_cat.index.name = f'Cluster_Label (Base: {col})'
            
            perfil_categorico_df = pd.concat([perfil_categorico_df, crosstab_cat.round(1)])

    print("\n[Tabela 3] Perfil Categórico (Proporção % de Gênero/Raça/Outros por Cluster):")
    # Para melhor visualização no console, ordenamos pelo rótulo do cluster
    print(perfil_categorico_df.sort_index())

    #======================
    relevancia_cluster = {}
    
    for cluster_label in range(n_clusters):
        relevancia = {}

        # 8a. Relevância Numérica/Ordinal (Magnitude do Desvio-Padrão)
        medias_cluster = perfil_numerico_transposto[cluster_label]
        # Filtra valores que têm desvio maior que 0.5 (um bom limiar)
        relevancia_numerica = medias_cluster[medias_cluster.abs() > 0.5].sort_values(ascending=False, key=abs)
        
        for feature, valor in relevancia_numerica.items():
            relevancia[feature] = f"{valor:.2f} σ" # Adiciona o desvio-padrão (magnitude)
            
        # 8b. Relevância Categórica (Magnitude da Desigualdade de Distribuição)
        for col_cat in COLUNAS_ROT_CAT:
            if col_cat in df.columns:
                # Obter a linha de interesse para o cluster
                try:
                    linha_cluster = perfil_categorico_df.loc[f'Cluster_Label (Base: {col_cat})', cluster_label].sort_values(ascending=False)
                    
                    # Identificar a categoria mais proeminente e sua proporção
                    categoria_mais_proeminente = linha_cluster.index[0]
                    proporcao = linha_cluster.iloc[0]
                    
                    # (Opcional) Comparar com a média geral (proporção no dataset total)
                    # Mas, para simplicidade, vamos destacar apenas a categoria mais forte do cluster
                    relevancia[f'{col_cat}: {categoria_mais_proeminente}'] = f"{proporcao:.1f}%"

                except KeyError:
                    # Ignora se a coluna categórica não foi processada corretamente
                    pass 

        # Ordenar e formatar o resultado
        relevancia_ordenada = {
            k: v for k, v in sorted(relevancia.items(), key=lambda item: (
                # Prioriza Númericas por |σ| e Categóricas por % (Decrescente)
                abs(float(item[1].split(' ')[0])) if 'σ' in item[1] else float(item[1].replace('%', '')),
                item[0]
            ), reverse=True)
        }
        relevancia_cluster[f'Cluster {cluster_label}'] = relevancia_ordenada
        
    print("\n[Etapa 8] RESUMO: Variáveis de Maior Relevância por Cluster (Ordem Decrescente):")
    for cluster, variaveis in relevancia_cluster.items():
        print(f"\n--- {cluster} ---")
        for k, v in variaveis.items():
            print(f"- {k}: {v}")

    # Retorno: Inclui o DataFrame categórico para possível uso externo
    
    return df, perfil_numerico, perfil_categorico

def avaliar_k_ideal(X, cols, max_k: int = 10):
    
    colunas_para_excluir_presentes = [col for col in cols if col in X.columns]
    X_final = X.drop(columns=colunas_para_excluir_presentes, errors='ignore')

    wcss = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    # Executa o K-Means para K de 2 até max_k
    for k in k_range:
        
        # 1. Treinamento
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        kmeans.fit(X_final)
        
        # 2. Métrica do Cotovelo (WCSS)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(X_final, kmeans.labels_)
        silhouette_scores.append(score)

        print(f"K={k}: WCSS={kmeans.inertia_:.2f}, Silhouette Score={score:.4f}")
        
    best_k_silhouette = max(silhouette_scores)
    k_for_best_silhouette = k_range[silhouette_scores.index(best_k_silhouette)]
    
    # 3. Plotagem dos Resultados
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Determinação do Número Ideal de Clusters (K)')

    # Gráfico 1: Método do Cotovelo (WCSS)
    ax[0].plot(k_range, wcss, marker='o', linestyle='-', color='darkblue')
    ax[0].set_title('Método do Cotovelo (WCSS)')
    ax[0].set_xlabel('Número de Clusters (K)')
    ax[0].set_ylabel('WCSS (Soma dos Quadrados Dentro do Cluster)')
    ax[0].grid(True, linestyle='--')
    ax[0].axvline(x=3, color='gray', linestyle='--', label='Cotovelo: K=3/4')
    ax[0].axvline(x=4, color='gray', linestyle='--')
    ax[0].legend()

    # Gráfico 2: Coeficiente de Silhueta
    ax[1].plot(k_range, silhouette_scores, marker='o', linestyle='-', color='darkred')
    ax[1].set_title('Coeficiente de Silhueta')
    ax[1].set_xlabel('Número de Clusters (K)')
    ax[1].set_ylabel('Score de Silhueta')
    ax[1].grid(True, linestyle='--')
    ax[1].axvline(x=k_for_best_silhouette, color='green', linestyle='-', linewidth=2, label=f'Melhor K: {k_for_best_silhouette}')
    ax[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    print(f"\nRecomendação Final: K={k_for_best_silhouette} (Maior Score de Silhueta).")

    return wcss, silhouette_scores

def ajuste(df):
    df_temp = df.copy()

    #REMOVER IGNORADO
    COLUNA_IGNORADA_COR_RACA = 'C009_9.0'
    if COLUNA_IGNORADA_COR_RACA in df_temp.columns:
    # Filtra: mantém apenas as linhas onde a coluna de ignorado (C009_9.0) é 0
        df_temp = df_temp[df_temp[COLUNA_IGNORADA_COR_RACA] == 0]
        # 3. EXCLUSÃO: Remover a coluna C009_9.0 do DataFrame
        df_temp = df_temp.drop(columns=[COLUNA_IGNORADA_COR_RACA], errors='ignore')

    #REMOVER DOENÇAS
    colunas_uf = [col for col in dataframe.columns if col.startswith('Q11006_')]
    colunas_para_excluir_presentes = [col for col in colunas_uf if col in df_temp.columns]
    df_temp = df_temp.drop(columns=colunas_para_excluir_presentes, errors='ignore')
    
    return df_temp

def preparar_transacoes(df):
    """
    Converte o DataFrame original em uma lista de transações categóricas.
    Cada linha vira um conjunto de itens (strings).
    """

    transacoes = []

    for _, row in df.iterrows():

        itens = []

        # ---- LOCAL ----
        mapa_local = {
            1.0: "Local=Capital",
            2.0: "Local=RegMetropolitana",
            3.0: "Local=AreaDesenv",
            4.0: "Local=Interior"
        }
        itens.append(mapa_local.get(row['V0031'], "Local=Desconhecido"))

        # ---- SEXO ----
        mapa_sexo = {1.0: "Sexo=Homem",
                      2.0: "Sexo=Mulher"}
        itens.append(mapa_sexo.get(row['C006'], "Sexo=Desconhecido"))

        # ---- IDADE ----
        if row['C008'] <= 25:
            itens.append("Idade=Jovem")
        elif row['C008'] <= 59:
            itens.append("Idade=Adulto")
        else:
            itens.append("Idade=Idoso")

        # ---- ETNIA ----
        mapa_cor = {
            1.0: "Raça=Branca",
            2.0: "Raça=Preta",
            3.0: "Raça=Amarela",
            4.0: "Raça=Parda",
            5.0: "Raça=Indigena"
        }
        itens.append(mapa_cor.get(row['C009'], "Raça=Desconhecida"))

        # ---- ESTADO CIVIL ----
        mapa_civil = {
            1.0: "Civil=Casado",
            2.0: "Civil=Divorciado",
            3.0: "Civil=Viúvo",
            4.0: "Civil=Solteiro"
        }
        itens.append(mapa_civil.get(row['C011'], "Civil=Outro"))

        # ---- INSTRUÇÃO ----
        if row['VDD004A'] <= 3:
            itens.append("Instrucao=Baixa")
        elif row['VDD004A'] <= 5:
            itens.append("Instrucao=Média")
        else:
            itens.append("Instrucao=Alta")

        # ---- RENDA ----
        if row['VDF004'] <= 3:
            itens.append("Renda=Baixa")
        elif row['VDF004'] == 4:
            itens.append("Renda=Média-Baixa")
        elif row['VDF004'] <= 6:
            itens.append("Renda=Média-Alta")
        else:
            itens.append("Renda=Alta")

        # ---- INTERNET ----
        mapa_internet = {1.0: "Internet=Sim",
                          2.0: "Internet=Não"}
        itens.append(mapa_internet.get(row['A01901'], "Internet=Desconhecido"))

        # ---- EXERCÍCIO ----
        mapa_ex = {1.0: "Exercicio=Sim",
                    2.0: "Exercicio=Não"}
        itens.append(mapa_ex.get(row['P034'], "Exercicio=Desconhecido"))

        # ---- ÁLCOOL ----
        alcool_inv = 4 - row['P027']   # Corrigido segundo sua inversão
        if alcool_inv == 0:
            itens.append("Alcool=Abstinente")
        elif alcool_inv == 1:
            itens.append("Alcool=Raramente")
        else:
            itens.append("Alcool=Frequente")

        # ---- TABACO ----
        if row['P050'] == 0:
            itens.append("Tabaco=Diario")
        elif row['P050'] == 1:
            itens.append("Tabaco=Esporadico")
        else:
            itens.append("Tabaco=NaoFuma")

        # ---- APOIO SOCIAL ----
        if row['M01401'] <= 1:
            itens.append("Apoio=Baixo")
        elif row['M01401'] <= 3:
            itens.append("Apoio=Médio")
        else:
            itens.append("Apoio=Alto")

        # ---- PHQ-9 ----
        itens.append(f"PHQ9={row['CAT_PHQ9']}")

        transacoes.append(itens)

    return transacoes

def treinar_regra_associacao(transacoes,
                             min_support=0.02,   # suporte do antecedente
                             min_confidence=0.30,
                             phq9_targets=None):

    # 1. Codificação
    te = TransactionEncoder()
    te_ary = te.fit(transacoes).transform(transacoes)
    df_bin = pd.DataFrame(te_ary, columns=te.columns_)

    # 2. Apriori: NÃO remove PHQ9 → deixa ele entrar mesmo raro
    itemsets = apriori(df_bin, min_support=min_support, use_colnames=True)

    # 3. Regras
    regras = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)

    # 4. Filtro pelo consequente (PHQ9)
    if phq9_targets:
        regras = regras[regras['consequents'].apply(
            lambda x: any(t in list(x) for t in phq9_targets)
        )]

    # 5. Ordenação
    regras = regras.sort_values(by=['lift', 'confidence', 'support'], ascending=False)

    return regras

colunas_chave = ['N010', 'N011', 'N012', 'N013', 'N014', 'N015', 'N016', 'N017', 'N018']

NOMINAIS_OHE = [
    'V0031',    #Tipo de área
    'C006',     # Sexo
    'C009',     # Cor ou raça
    'C011',     # Estado civil
    'A01901',   # Acesso à Internet
    'Q00201',   # Variável binária (Sim/Não)
    'Q03001',   # Variável binária (Sim/Não)
    'Q06306',   # Variável binária (Sim/Não)
    'Q084',     # Variável binária (Sim/Não)
    'Q132',     # Variável binária (Sim/Não)
    'P034',     # Variável binária (Sim/Não)
]

ORDINAIS_MAP = [
    'VDD004A',  # Escolaridade (1=Sem instrução, 7=Superior completo) - Ordem OK
    'VDF004',   # Renda domiciliar per capita (faixas em salário minimo) - Ordem OK
    'J00101',   # Autoavaliação da saúde (1=Muito Bom, 5=Muito Ruim) - Precisa de INVERSÃO
    'P027',     # Frequência de consumo de álcool(1-Não bebe; 3-Recorrente) - Precisa de inversão
    'P050',     # Frequência de consumo de Tabaco(1-Diariamente; 3-Não fuma) - Ordem OK
    'M01401',   # Apoio Social Familiar (1-Ninguém;3-Três ou mais pessoas) - Ordem OK
]

NUMERICAS = [
    'C008',     # Idade
    'E01602',   # Rendimento bruto mensal
]

ORD_NUM = [
    'VDD004A',  # Escolaridade (1=Sem instrução, 7=Superior completo) - Ordem OK
    'VDF004',   # Renda domiciliar per capita (faixas em salário minimo) - Ordem OK
    'J00101',   # Autoavaliação da saúde (1=Muito Bom, 5=Muito Ruim) - Precisa de INVERSÃO
    'P027',     # Frequência de consumo de álcool(1-Não bebe; 3-Recorrente) - Precisa de inversão
    'P050',     # Frequência de consumo de Tabaco(1-Diariamente; 3-Não fuma) - Ordem OK
    'M01401',   # Apoio Social Familiar (1-Ninguém;3-Três ou mais pessoas) - Ordem OK
    'C008',     # Idade
    'E01602',   # Rendimento bruto mensal
]

ORDINAIS_RA = [
    'VDD004A',  # Escolaridade (1=Sem instrução, 7=Superior completo) - Ordem OK
    'VDF004',   # Renda domiciliar per capita (faixas em salário minimo) - Ordem OK
    'J00101',   # Autoavaliação da saúde (1=Muito Bom, 5=Muito Ruim) - Precisa de INVERSÃO
    'P027',     # Frequência de consumo de álcool(1-Não bebe; 3-Recorrente) - Precisa de inversão
    'P050',     # Frequência de consumo de Tabaco(1-Diariamente; 3-Não fuma) - Ordem OK
    'M01401',   # Apoio Social Familiar (1-Ninguém;3-Três ou mais pessoas) - Ordem OK
]


COLUNAS_PARA_EXCLUIR = [
        'CAT_PHQ9', 'SCR_PHQ9',
        'V03502', 'Q092', 'Q109',
        'N010', 'N011', 'N012', 'N013', 'N014', 'N015', 'N016', 'N017', 'N018',
        'V001'
    ]

COLUNAS_ROTULADA = {'C008': 'Idade',
                    'E01602': 'Renda pessoal',
                    'VDD004A': 'Nível de instrução',
                    'VDF004': "Renda domiciliar",
                    'J00101': 'Autoavaliação de saúde mental',
                    'P027': 'Consumo de bebida alcóolica',
                    'P050': 'Consumo de produto com tabaco',
                    'M01401': 'Rede de apoio'}

COLUNAS_ROT_CAT = {
    'V0031': 'Tipo de área',    # 1- CAPITAL; 2-REGIÃO METROPOLITANA; 3-REGIÃO INTEGRADA DE DESENVOLVIMENTO; 4-INTERIOR
    'C006': 'Sexo',     # 1- HOMEM; 2- MULHER
    'C009': 'Etnia',     # 1- BRANCA; 2- PRETA; 3- AMARELA; 4- PARDA; 5-INDIGENA
    'C011': 'Estado civil',     # 1- CASADO; 2- DIVORCIADO; 3-VIÚVO; 4-SOLTEIRO
    'A01901': 'Acesso à internet',   # 1- Sim; 2- Não
    'Q00201': 'Pressão alta?',   # 1- Sim; 2- Não
    'Q03001': 'Diabético?',   # 1- Sim; 2- Não
    'Q06306': 'Cardíaco?',   # 1- Sim; 2- Não
    'Q084': 'Problemas na coluna?',     # 1- Sim; 2- Não
    'Q132': 'Problemas para dormir?',     # 1- Sim; 2- Não
    'P034': 'Pratica exercício?',     # 1- Sim; 2- Não
}

dataframe = pd.read_csv(nome_arquivo)

'''
Pré-processamento

df_temp = novo_df_registro_limpo(dataframe)
print(df_temp)
df_temp = score_PHQ9(df_temp, colunas_chave)
df_temp = categ_PHQ9(df_temp)
df_temp = OneHotEnconding(df_temp, NOMINAIS_OHE)
df_temp = pre_process_ordinais(df_temp, ORDINAIS_MAP)
df_temp = preprocess_scaling(df_temp, ORD_NUM)
df_temp = ajuste(df_temp)
df_temp.to_csv('basedados_padronizada.csv', index=False)
'''
#df_final_com_rotulos, perfil_numerico, perfil_categorico = realizar_clusterizacao_e_plotagem_final(dataframe, COLUNAS_PARA_EXCLUIR, 8, n_clusters=2)
#df_final_com_rotulos, perfil_numerico, perfil_categorico = realizar_clusterizacao_e_plotagem_sempca(dataframe, COLUNAS_PARA_EXCLUIR, n_clusters=5)
#df_final_com_rotulos, perfil_numerico, perfil_categorico = realizar_clusterizacao_e_plotagem_final(dataframe, COLUNAS_PARA_EXCLUIR, 8, n_clusters=9)
#df_final_com_rotulos, perfil_numerico, perfil_categorico = realizar_clusterizacao_e_plotagem_final(dataframe, COLUNAS_PARA_EXCLUIR, 8, n_clusters=5)
#df_final_com_rotulos, perfil_numerico, perfil_categorico = realizar_clusterizacao_e_plotagem_final(dataframe, COLUNAS_PARA_EXCLUIR, 8, n_clusters=15)
#df_final_com_rotulos, perfil_numerico, perfil_categorico = realizar_clusterizacao_e_plotagem_sempca(dataframe, COLUNAS_PARA_EXCLUIR, n_clusters=15)
