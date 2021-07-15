import streamlit as st
import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt


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
    pl_df = phras_df[(phras_df.label == label.lower()) & (phras_df.prep == prep)]
    hosts = pl_df.host_lemma.value_counts().head(10)
    dependants = pl_df.dependant_lemma.value_counts().head(10)
    
    if st.checkbox('Показать основных хозяев'):
        hosts_fig = utils.show_hbar(
            hosts.index,
            hosts.values,
            title='Относительная частота хозяев синтаксемы')
        st.write(hosts_fig)

    if st.checkbox('Показать основных слуг'):
        deps_fig = utils.show_hbar(
            dependants.index,
            dependants.values,
            title='Относительная частота слуг синтаксемы')
        st.write(deps_fig)


# fig, ax = plt.subplots()

# # Example data
# people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
# y_pos = np.arange(len(people))
# performance = 3 + 10 * np.random.rand(len(people))
# error = np.random.rand(len(people))

# ax.barh(y_pos, performance, xerr=error, align='center')
# ax.set_yticks(y_pos)
# ax.set_yticklabels(people)
# ax.invert_yaxis()  # labels read top-to-bottom
# ax.set_xlabel('Performance')
# ax.set_title('How fast do you want to go today?')

# st.write(fig)

