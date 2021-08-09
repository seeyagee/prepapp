import streamlit as st
import pandas as pd
import utils
from utils import hrule
import models
import json


@st.cache(allow_output_mutation=True)
def load_models():
    extractor = models.Extractor(
        model='static/ru_core_news_sm/ru_core_news_sm-3.0.0')
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
    base_prep = json.loads(
            open('static/base_prep.json').read())
    label_definitions = json.loads(
            open('static/definitions.json').read())
    semantic_df = {
            'Синонимы': pd.read_csv(
                'static/synonyms.csv'),
            'Антонимы': pd.read_csv(
                'static/antonyms.csv')}
    return phras_df, synt_df, label_definitions, \
           base_prep, semantic_df

st.title('Предложные конструкции в русском языке')
st.write('Место для вступительного слова по проекту')

phras_df, synt_df, label_definitions, base_prep, semantic_df = load_data()

all_preps = sorted(base_prep.keys(), key=len)

st.header(':book:')
prep = st.selectbox(
    'Выберите предлог, чтобы узнать больше',
    ['']+all_preps)

if prep:
    prep_df = synt_df[synt_df.prep == prep]
    prep_labels = sorted(prep_df.label.unique())

    st.markdown(f"""
            ##\n## __Паспорт предлога__ `{prep.upper()}`\n##""")

    variants = ", ".join(
                    [f'`{p.upper()}`' for p in  base_prep[prep]['variants']])
    variants = variants or 'Нет'
    st.markdown(f"""
            __Варианты:__ {variants+hrule}""")

    st.markdown(f"""
            __Тип__: `{base_prep[prep]['kind']}`{hrule}""")

    st.markdown(f"""
            __Падежи:__ {", ".join(
                    [f'`{c}`' for c in base_prep[prep]['case']])+hrule}""")

    st.markdown(f"""
            __Значения:__{", ".join(
                    [f'`{l}`' for l in prep_df.label.unique()])}\n##""")

    if len(prep_df) and st.checkbox('Подробнее о значениях'):

        for l in prep_labels:
            with st.beta_expander(f'{l}'):

                short_df = synt_df[(synt_df.prep == prep) & (synt_df.label == l)]
                definitions = label_definitions[l].split(';')

                for d in definitions:
                    st.write(f'- {d.capitalize()}')
                st.write(
                    '###\n## Примеры: ')

                for _, row in short_df.iterrows():
                    st.write(
                        f'### {row.case} падеж:')

                    for ex in row.examples.split(','):
                        st.write(f'*{ex.strip()}*\n')

    motive = base_prep[prep].get('motive', 'Нет')
    if motive != 'Нет':
        motive = f'`{motive}`'
    st.markdown(f"""
            {hrule}\n__Мотивирующее слово__: {motive}""")

    for ent in 'Синонимы', 'Антонимы':
        df = semantic_df[ent]
        df = df[df['prep_1'] == prep]
        if not len(df):
            st.markdown(f"""
                {hrule}\n__{ent}__:  Нет""")
            continue
        st.markdown(f"""
            {hrule}\n__{ent}__:\n\n""")
        with st.beta_expander(f'Показать {ent.lower()}:'):
            for label in df.label.unique():
                sem_group = df[df.label == label]
                st.markdown(f"""
                        \n- __{label}__\n""")
                for _, row in sem_group.iterrows():
                    st.markdown(f"""
                        `{row['prep_2'].upper()}` + `{row['case_2'].capitalize()}` ⸻ \
                        `{row['prep_1'].upper()}` + `{row['case_1'].capitalize()}`""")

    idioms = base_prep[prep]['idiom']
    if not len(idioms):
        st.markdown(f"""
                {hrule}\n__Идиомы__:  Нет""")
    if len(base_prep[prep]['idiom']):
        st.markdown(f"""
                {hrule}\n__Идиомы__:\n\n""")
        with st.beta_expander(f'Показать идиомы:'):
            for idiom in base_prep[prep]['idiom']:
                st.markdown(f'- {idiom.capitalize()}')

    st.markdown(f"""
            {hrule}\n__Источники:__\n\n""")
    with st.beta_expander(f'Показать источники:'):
        for source in base_prep[prep]['source']:
            st.markdown(f'- {source}')
    st.markdown(hrule)


st.header(':mag_right:')
with st.beta_expander(
            'Сформировать запрос в банк предложных конструкций:'):
    query = {}

    query_values = (
        'prep', 'label', 'dependant_lemma', 'dependant_case',
        'dependant_pos', 'host_lemma', 'host_pos')

    for col in query_values:

        query[col] = st.multiselect(
                    col,
                    sorted(phras_df[col].unique().tolist()))
    
    if st.checkbox(
            'Показать таблицу'):

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
text = st.text_area(label='Извлечь конструкции из своего текста:',
                    max_chars=400,
                    help='Введите текст. Например, "Мероприятие в честь выпуска выпадает на субботу"')

extractor, classifier = load_models()

if st.button('Извлечь') or text:
    pphrase_gen = extractor.parse(text)
    st.markdown('#### Найденные конструкции:')
    for elem in pphrase_gen:
        text = utils.preprocess(elem)
        label = classifier.predict(text)[0]

        # TODO needs model retraining to remove this hack
        if label == 'каузатор':
            label = 'каузатив'
        elif label == 'квалитатив':
            label = 'квалификатив'

        pphrase = elem['prep'].lower() + ' ' + elem['dependant']
        if elem['host'] is not None:
            pphrase = elem['host'] + ' ' + pphrase
        pphrase = (pphrase[:1].upper() + pphrase[1:]).strip()
        
        st.markdown(f"{pphrase} ⸻ `{label}`")
