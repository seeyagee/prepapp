import streamlit as st
import pandas as pd
import numpy as np
import utils


st.title('Предложные конструкции в русском языке')
st.write('Место для вступительного слова по проекту')

phras_df = pd.read_csv('static/prep_phrases_gold.csv')
synt_df = pd.read_csv('static/syntaxemes.csv')

all_preps = sorted(phras_df.prep.unique(), key=len)

prep = st.selectbox(
    'Выберите предлог, чтобы узнать больше',
    ['']+all_preps)

if not prep:
    st.stop()

prep_df = synt_df[synt_df.prep == prep]
prep_labels = sorted(prep_df.label.unique())

st.markdown(f'#### Значения предлога *{prep.upper()}*:')
for l in prep_labels:
    with st.beta_expander(f'{l}'):

        short_df = synt_df[(synt_df.prep == prep) & (synt_df.label == l)]
        definitions = short_df['definition'].iloc[0].split(';')
        for d in definitions:
            st.write(f"""
            - {d}
            """)
        st.write("###\n## Примеры: ")
        for _, row in short_df.iterrows():

            st.write(f"### {row['case']} падеж:")
            for ex in row['examples'].split(','):
                st.write(f"*{ex.strip()}*\n")


label = st.selectbox(
    'Выберите значение',
    [''] + prep_labels) 

if label:
    st.write(f'{label}')

