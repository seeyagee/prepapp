import streamlit as st
import pandas as pd
import utils
import models


@st.cache(allow_output_mutation=True)
def load_models():
    extractor = models.Extractor(
        model='ru_core_news_sm')
    classifier = models.Classifier(
        model='static/classifier.pkl',
        vectorizer='static/ft_freqprune_100K_20K_pq_100.bin')
    return extractor, classifier

@st.cache()
def load_data():
    phras_df = pd.read_csv(
    'static/prep_phrases_gold.csv')
    synt_df = pd.read_csv(
    'static/syntaxemes.csv')
    return phras_df, synt_df

st.title('Предложные конструкции в русском языке')
st.write('Место для вступительного слова по проекту')

phras_df, synt_df = load_data()

all_preps = sorted(phras_df.prep.unique(), key=len)

st.header(':book:')
prep = st.selectbox(
    'Выберите предлог, чтобы узнать больше',
    ['']+all_preps)

if prep:
    prep_df = synt_df[synt_df.prep == prep]
    prep_labels = sorted(prep_df.label.unique())


    st.markdown(f'#### Значения предлога *{prep.upper()}*:')
    for l in prep_labels:
        with st.beta_expander(f'{l}'):

            short_df = synt_df[(synt_df.prep == prep) & (synt_df.label == l)]
            definitions = short_df.definition.iloc[0].split(';')
            
            for d in definitions:
                st.write(f'- {d}')
            st.write('###\n## Примеры: ')
            
            for _, row in short_df.iterrows():
                st.write(f'### {row.case} падеж:')
            
                for ex in row.examples.split(','):
                    st.write(f'*{ex.strip()}*\n')

    label = st.selectbox(
        'Выберите значение',
        [''] + prep_labels) 

    if label:
        pl_df = phras_df[(phras_df.label == label.lower()) & (phras_df.prep == prep)]
        hosts = pl_df.host_lemma.value_counts().head(10)
        dependants = pl_df.dependant_lemma.value_counts().head(10)
        
        if st.checkbox('Показать хозяев'):
            hosts_fig = utils.show_hbar(
                hosts.index,
                hosts.values,
                title='Относительная частота хозяев синтаксемы')
            st.write(hosts_fig)

        if st.checkbox('Показать слуг'):
            deps_fig = utils.show_hbar(
                dependants.index,
                dependants.values,
                title='Относительная частота слуг синтаксемы')
            st.write(deps_fig)


st.header(':mag_right:')
with st.beta_expander('Сформировать запрос в банк предложных конструкций:'):
    query = {}

    query_values = (
        'prep', 'host_lemma', 'dependant_lemma',
        'dependant_case', 'host_pos', 'dependant_pos', 'label')

    for col in query_values:

        query[col] = st.multiselect(
                    col,
                    phras_df[col].unique().tolist())
    
    if st.checkbox('Показать таблицу'):

        query  = ' and '.join(
            f'{key} in {val}' for key, val in query.items()
            if len(val))
        out_df = phras_df.query(query) if query else phras_df
        st.write(out_df[['phrase', 'label', 'host', 'prep', 'dependant']])

        tmp_link = utils.get_table_download_button(
            out_df,
            'prep_phrases_query.csv',
            'Cкачать')
        st.markdown(tmp_link, unsafe_allow_html=True)


st.header(':pencil:')
text = st.text_area('Извлечь конструкции из текста:')

extractor, classifier = load_models()

if text:
    pphrase_gen = extractor.parse(text)
    for elem in pphrase_gen:
        text = utils.preprocess(elem)
        label = classifier.predict(text)[0]

        pphrase = elem['prep'].lower() + ' ' + elem['dependant']
        if elem['host'] is not None:
            pphrase = elem['host'] + ' ' + pphrase

        pphrase = (pphrase[:1].upper() + pphrase[1:]).strip()
        
        st.write(pphrase, '-----', label)
