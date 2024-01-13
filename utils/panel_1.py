import pandas as pd
import numpy as np
import plotly.graph_objects as go

import plotly.express as px


def get_histogram(df):
    # df = px.data.tips()
    # Here we use a column with categorical data
    fig = px.histogram(df, x="Topic").update_xaxes(categoryorder='total descending')

    # fig = px.histogram(df, x="Topic", histnorm='percent').update_xaxes(categoryorder='total descending')
    # fig.update_traces(hovertemplate='count=%{count} <br>percent = %{y:.2f}<extra>%{x}</extra>')

    # fig.show()
    return fig

