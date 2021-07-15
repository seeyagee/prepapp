import matplotlib.pyplot as plt
import numpy as np
import base64
import re
import uuid


def show_hbar(names, values, title, kind='relative'):

    if kind == 'relative':
        values = (values/ sum(values)).round(2)

    fig, ax = plt.subplots()

    y_pos = np.arange(len(values))

    ax.barh(y_pos, values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_title(title)

    return fig


def get_table_download_button(df, filename, linktext):

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" id="b0" download="{filename}">&#128190  {linktext}</a><br></br>'

    custom_css = f""" 
        <style>
            #b0 {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(246, 51, 102);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #b0:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #b0:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """


    return custom_css + href
