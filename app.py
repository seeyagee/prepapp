import streamlit as st
import pandas as pd
import numpy as np


# st.title('Предложные конструкции в русском языке')
# st.write('')
# df = pd.read_csv('static/prep_phrases_gold.csv.zip', compression='zip')
# df = df[['phrase', 'label']]

# # st.write("## Data", df)

# genre = st.radio(
#     "What's your favorite movie genre",
#     ('Comedy', 'Drama', 'Documentary'))
# if genre == 'Comedy':
#     st.write('You selected comedy.')
# else:
#     st.write("You didn't select comedy.")


# options = st.multiselect(
#      'What are your favorite colors',
#     ['Blue' for _ in range(100)]+['Yellow']+ ['Red']	)

# st.write('You selected:', options)


if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data