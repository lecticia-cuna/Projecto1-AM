import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


st.title("Predição de Câncer de Mama em Moçambique")
tab1, tab2, tab3 = st.tabs(["Informações sobre o Câncer de Mama", "Análise de Dados", "Modelo e Predição"])

# abaa 1
with tab1:
    st.header("Câncer de Mama em Moçambique")
    st.write("""
        O câncer de mama é a segunda causa mais comum de morte relacionada com o câncer entre as mulheres em Moçambique.
        Nos últimos 5 anos, o cancro da mama foi o cancro mais diagnosticado nas mulheres, afectando mais
        a faixa etária dos 30 aos 59 anos de idade e foi responsável por 8% das mortes por todas as causas de cancro.
        Muitas mulheres enfrentam diagnósticos tardios devido à falta de acesso a serviços de saúde especializados e 
        a ferramentas de diagnóstico avançadas.
    """)


    st.image("C:\\Users\\Dell\\Desktop\\download.jfif", caption="Câncer de Mama", width=700)

    # Problemas identificados
    st.subheader("Problemas")
    st.write("""
        - Falta de deteção atempada  
        - Infra-estruturas de saúde limitadas  
        - Lacunas na sensibilização e na educação
    """)
    st.image("C:\\Users\\Dell\\Desktop\\images.png", caption="Câncer de Mama", width=700)

    # Solução proposta
    st.subheader("Solução")
    st.write("""
        Para resolver os problemas que forma descritos anteriormente, sera utilizado um modelo treinado com algoritmo de 
        aprendizagem de maquina para detectar câncer de mama.
        Esta solução oferece várias vantagens, como:
        - Diagnósticos mais rápidos e precisos
        - Auxílio em regiões com poucos especialistas
        - Melhoria da taxa de deteção precoce, resultando em melhores tratamentos e recuperação.
    """)


    st.image("C:\\Users\\Dell\\Desktop\\images.jfif", caption="Câncer de Mama", width=700)
# aba 2
with tab2:
    st.header("Análise de Dados e Machine Learning")

    # Carregar os dados
    df = pd.read_csv("breastcancer.csv")
    st.subheader("1. Pré-processamento dos Dados")
    st.write("""O primeiro passo foi realizar o pré-processamento dos dados.""")

    # Exibir os primeiros dados do dataset
    st.subheader("1. Dados Iniciais")
    st.write("Primeiros dados do dataset:")
    st.dataframe(df.head())

    # Informações gerais sobre os dados
    st.subheader("2. Informações gerais sobre o dataset")
    st.write(df.describe())

    st.write("""  
        A coluna 'diagnosis' foi convertida de valores categóricos para numéricos, onde M (Maligno) foi mapeado para 1,
         e B (Benigno) para 0.
    """)

    # Transformar a coluna 'diagnosis'
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])

    st.write("Dados após a transformação:")
    st.dataframe(df[['diagnosis']].head())

    # Dividir os dados
    X = df.drop(columns=['diagnosis', 'id'])
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write("Os dados foram divididos em treino e teste.")

    # Exploração de Dados
    st.subheader("3. Exploração de Dados")
    st.write("Foi criada uma matriz de correlação para explorar as relações entre os atributos.")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)

    # Distribuição de diagnósticos
    st.write("A distribuição dos diagnósticos foi visualizada:")
    plt.figure(figsize=(6, 4))
    sns.countplot(x='diagnosis', data=df)
    st.pyplot(plt)

    # Algoritmos de Machine Learning
    st.subheader("4. Algoritmos de Machine Learning")
    st.write("""
        Para a classificação do cancer de mama, foi utilizado o algoritmo Random Forest.
        O Random Forest foi escolhido por suas vantagens, incluindo alta precisão, robustez contra overfittinge boa performance
        o que ira facilitar no processo de detecção do câncer.
    """)

    # Treinar o modelo
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    st.write("O modelo foi treinado com sucesso.")

    # Avaliação do modelo
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write("Abaixo está o relatório de classificação do modelo:")
    st.dataframe(report_df)

    # Importância dos atributos
    st.subheader("5. Importância dos Atributos")
    importances = model.feature_importances_
    feat_importances = pd.Series(importances, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    st.pyplot(plt)

# a ba 3
with tab3:
    st.header("Funcionalidade do Modelo de Predição")

    # Carregar o arquivo CSV para teste do modelo
    uploaded_file = st.file_uploader("Faça o upload de um arquivo CSV com os dados de teste para prever", type=["csv"])

    if uploaded_file is not None:
        # Carregar os dados do arquivo CSV
        test_data = pd.read_csv(uploaded_file)

        st.write("Dados carregados com sucesso:")
        st.dataframe(test_data.head())

        # Verificar se o modelo foi treinado
        if 'diagnosis' in test_data.columns:
            # Prever com base nos dados fornecidos
            X_test_new = test_data.drop(columns=['diagnosis', 'id'], errors='ignore')
            y_pred_new = model.predict(X_test_new)

            st.write("Previsões realizadas com sucesso:")
            test_data['Predição'] = y_pred_new
            st.dataframe(test_data)

            # Exibir diagnóstico com base nas previsões
            st.write("""
                **Nota**:  
                1 = Maligno (Câncer)  
                0 = Benigno (Sem câncer)
            """)
