import plotly.express as px
import pandas as pd

# update/add code below ...
df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')

def survival_demographics():
    # Use the global df loaded in cell 2
    age_bins = [0, 12, 19, 59, float('inf')]
    age_labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)
    df['age_group'] = df['age_group'].astype('category')

    grouped = df.groupby(['Pclass', 'Sex', 'age_group'])
    summary = grouped.agg(
        n_passengers=('PassengerId', 'count'),
        n_survivors=('Survived', 'sum')
    ).reset_index()
    summary['survival_rate'] = summary['n_survivors'] / summary['n_passengers']
    summary = summary.sort_values(by=['Pclass', 'Sex', 'age_group']).reset_index(drop=True)
    return summary

def visualize_demographic():
    summary_df = survival_demographics()
    # Highlight adult women in first class
    highlight = (
        (summary_df['Pclass'] == 1) &
        (summary_df['Sex'] == 'female') &
        (summary_df['age_group'] == 'Adult')
    )
    summary_df['highlight'] = highlight

    fig = px.bar(
        summary_df,
        x='age_group',
        y='survival_rate',
        color='Sex',
        facet_col='Pclass',
        barmode='group',
        title='Survival Rate by Class, Sex, and Age Group',
        color_discrete_map={'male': 'steelblue', 'female': 'salmon'},
        hover_data=['n_passengers', 'n_survivors']
    )

    fig.update_traces(marker_line_width=1.5, marker_line_color='black')
    fig.update_layout(showlegend=True)

    return fig