import streamlit as st
import pandas as pd
import numpy as np
import utils


st.title('Предложные конструкции в русском языке')
st.write('Место для вступительного слова по проекту')

df = pd.read_csv('static/prep_phrases_gold.csv.zip', compression='zip')

all_preps = sorted(df.prep.unique(), key=len)

prep = st.selectbox(
	'Выберите предлог, чтобы узнать больше',
	['']+all_preps)

if not prep:
	st.stop()

prep_df = df[df.prep == prep]
prep_labels = sorted(prep_df.label.unique())

st.markdown(f'#### Значения предлога *{prep.upper()}*:')
for l in prep_labels:
	with st.beta_expander(f'{l.capitalize()}'):
		st.write("""
			Дефиниция + примеры в контесте предлога
			""")

label = st.selectbox(
	'Выберите значение',
	[''] + prep_labels) 

if label:
	st.write(f'{label}')