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

def family_groups():
    # Use the global df loaded in cell 2
    
    # Calculate family size
    df['family_size'] = df['SibSp'] + df['Parch'] + 1

    # Group by family size and class
    grouped = df.groupby(['family_size', 'Pclass'])

    # Aggregate statistics
    summary = grouped.agg(
        n_passengers=('PassengerId', 'count'),
        avg_fare=('Fare', 'mean'),
        min_fare=('Fare', 'min'),
        max_fare=('Fare', 'max')
    ).reset_index()

    # Sort for readability
    summary = summary.sort_values(by=['Pclass', 'family_size']).reset_index(drop=True)

    return summary

def last_names():
    # Use the global df loaded in cell 2

    # Extract last names
    df['last_name'] = df['Name'].str.extract(r'^([^,]+),')[0]

    # Count occurrences
    last_name_counts = df['last_name'].value_counts()

    return last_name_counts

def visualize_families():
    # Get summary data
    summary_df = family_groups()

    # Create scatter plot
    fig = px.scatter(
        summary_df,
        x='family_size',
        y='avg_fare',
        color='Pclass',
        size='n_passengers',
        title='Average Fare by Family Size and Class',
        labels={'family_size': 'Family Size', 'avg_fare': 'Average Fare'},
        hover_data=['min_fare', 'max_fare']
    )

    fig.update_layout(showlegend=True)
    return fig