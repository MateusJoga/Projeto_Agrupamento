# 🧠 Análise de Sintomas Depressivos com K-Means e Regras de Associação (Apriori)

Este projeto realiza uma análise exploratória e preditiva da **saúde mental da população brasileira**, utilizando os dados da **Pesquisa Nacional de Saúde (PNS - 2019)**.  
O objetivo central é identificar **perfis sociais associados a sintomas depressivos**, modelados através do **PHQ-9**, uma métrica clínica amplamente usada para rastreamento de depressão.

---

## 🎯 Objetivo do Projeto

1. Detectar perfis de risco a partir de variáveis socioeconômicas e comportamentais.
2. Mapear relações importantes entre fatores sociais e sintomas depressivos.
3. Utilizar técnicas de aprendizado não supervisionado para revelar padrões não visíveis a olho nu.
4. Produzir regras de associação interpretáveis que complementem as descobertas dos clusters.

---

## 📊 Técnicas Utilizadas

### 🔹 1. K-Means + PCA
- Redução de dimensionalidade com **PCA** (8 componentes principais).
- Agrupamento com **K-Means** (testes variando K=2 a K=15).
- Teste do modelo usando **Coeficiente de Silhouette** e **Método de cotovelo**.
- Interpretação dos clusters com base em renda, escolaridade, exercício, etnia, idade etc.

### 🔹 2. Apriori (Regras de Associação)
- Transformação completa das variáveis em formato transacional (One-Hot).
- Suporte reduzido em **2%**.
- Métricas analisadas: Support, Confidence, Lift.
- Filtro para regras relacionadas às categorias mais altas do PHQ-9.

---

## 🗂️ Dados Utilizados
- Base: **PNS 2019 – IBGE**
- Variáveis demográficas, socioeconômicas, comportamentais e de saúde.
- Total após tratamentos: **47.346 registros**

---

## 🔍 Principais Resultados

### ⭐ 1. Perfis Vulneráveis Identificados (K-Means)
- Baixa renda + baixa escolaridade → maior prevalência de PHQ9 ≥ 10.
- Sedentarismo como fator comportamental mais consistente.
- Etnia com peso significativo.
- Renda + Escolaridade funcionam como fatores protetivos.

### ⭐ 2. Regras de Associação Significativas
Exemplos de associações relevantes:
- Sexo = Mulher → maior chance de PHQ9 Moderado.
- Não fumar + ser mulher → associação com PHQ9 Moderado.
- Baixa renda + baixa instrução → ligação com sintomas depressivos.
- Consumo frequente de álcool → associado ao PHQ9 Moderado.
- Alta escolaridade → risco reduzido.

---

## 🏁 Conclusão

O projeto demonstra que a combinação entre **clustering + regras de associação** é eficaz para identificar grupos vulneráveis, compreender fatores contextuais da depressão e gerar insights interpretáveis para políticas públicas.

