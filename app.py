import streamlit as st
import pandas as pd
import utils
from utils import hrule
import models
import json
import numpy as np

st.set_page_config(page_title='Предлоги в русском языке')


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
            'static/prep_phrases_db.csv')
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


@st.cache()
def load_configs():
    formatter = json.loads(
            open('static/format.json').read())
    default = json.loads(
            open('static/default.json').read())
    return formatter, default


phras_df, synt_df, label_definitions, base_prep, semantic_df = load_data()
formatter, default = load_configs()

all_preps = sorted(base_prep.keys(), key=len)

st.title('Предложные конструкции в русском языке')
st.markdown(formatter['intro'])


st.header(':book:')
prep = st.selectbox(
    'Выберите предлог из списка, чтобы просмотреть его паспорт',
    all_preps)

if prep:
    prep_df = synt_df[synt_df.prep == prep]
    prep_labels = sorted(prep_df.label.unique())

    st.markdown(f"""
            ##\n## __Паспорт предлога__ `{prep.upper()}`\n##""")

    variants = ", ".join(
                    [f'`{p.upper()}`' for p in base_prep[prep]['variants']])
    variants = variants or 'Нет'
    st.markdown(f"""
            __Варианты:__ {variants+hrule}""")

    st.markdown(f"""
            __Тип__: `{base_prep[prep]['kind']}`{hrule}""")

    st.markdown(f"""
            __Падежи:__ {", ".join(
                    [f'`{formatter[c.lower()].capitalize()}`' for c in base_prep[prep]['case']])+hrule}""")

    st.markdown(f"""
            __Значения:__{", ".join(
                    [f'`{l.capitalize()}`' for l in prep_df.label.unique()])}\n##""")

    if len(prep_df) and st.checkbox('Подробнее о значениях'):

        for l in prep_labels:
            with st.expander(f'{l}'):

                short_df = prep_df[(prep_df.label == l)]
                definitions = label_definitions[l].split(';')

                for d in definitions:
                    st.write(f'- {d.lower()}')
                st.write(
                    '###\n## Примеры: ')

                for _, row in short_df.iterrows():
                    st.write(
                        f'### {formatter[row.case.lower()].capitalize()} падеж:')

                    for ex in row.examples.split(','):
                        st.write(f'*{ex.strip().capitalize()}*\n')

    #Тут про стиль предлога

    typical_style = base_prep[prep]['style']['typical']
    non_typical_style = base_prep[prep]['style']['non-typical']
    if typical_style or non_typical_style:
        if typical_style[0] == 'общеупотребимый':
            st.markdown(f"""
                {hrule}\n__Стиль__: `Общеупотребимый`""")
        else:
            st.markdown(f"""
                {hrule}\n__Стиль__: """)
            st.markdown(f"""
                    Типичный: {", ".join([f'`{s}`' for s in typical_style])}""")
            st.markdown(f"""
                    Нетипичный: {", ".join([f'`{s}`' for s in non_typical_style])}""")

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
        with st.expander(f'Показать'):
            for label in df.label.unique():
                if label is np.nan:
                    continue
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
        with st.expander('Показать'):
            for idiom in base_prep[prep]['idiom']:
                st.markdown(f'- {idiom.capitalize()}')

    st.markdown(f"""
            {hrule}\n__Источники:__\n\n""")
    with st.expander('Показать'):
        for source in base_prep[prep]['source']:
            st.markdown(f'- {formatter.get(source, source)}')
    st.markdown(hrule)

st.markdown(formatter['goals_1'])
st.markdown(formatter['goals_2'])
st.markdown(formatter['query'])

st.header(':mag_right:')
with st.expander(
            'Нажмите, чтобы сформировать запрос в Банк предложных конструкций'):
    example = st.checkbox('Пример')
    with st.form(key='query'):
        query = {}

        for col in default['query_values']:
            d = default[col] if example else None
            query[col] = st.multiselect(
                    label=formatter.get(col, col),
                    options=sorted(phras_df[col].unique().tolist()),
                    default=d
                    )
        submit = st.form_submit_button(label='Искать')

    if submit:
        query = ' and '.join(
            f'{key} in {val}' for key, val in query.items()
            if len(val))
        out_df = phras_df.query(query) if query else phras_df
        st.write(out_df[['phrase', 'label', 'host', 'prep', 'dependant', 'corpus']])

        tmp_link = utils.get_table_download_button(
            out_df,
            'prep_phrases_query.csv',
            'Cкачать')
        st.markdown(tmp_link, unsafe_allow_html=True)

st.markdown(formatter['text_area'])

st.header(':pencil:')
with st.form(key='extraction'):
    text = st.text_area(label='Введите текст, чтобы извлечь предложные конструкции',
                        value=default["text_value"],
                        max_chars=default["text_max_chars"],
                        help=f'Введите текст длиной до {default["text_max_chars"]} символов и нажмите "Извлечь", чтобы получить список предложных конструкций из текста и их значений.',
                        height=200)
    extract = st.form_submit_button(label='Извлечь')
    if extract:
        extractor, classifier = load_models()
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

st.markdown(formatter['outro'])
st.markdown(formatter['credits'])
