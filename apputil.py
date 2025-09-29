import plotly.express as px
import pandas as pd
import numpy as np

# update/add code below ...
df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')
# provide a lowercase alias used by tests
if 'Pclass' in df.columns and 'pclass' not in df.columns:
    df['pclass'] = df['Pclass']


def survival_demographics():
    '''
    Function to summarize survival demographics based on grouping needs specified.

    Parameters: none

    Returns a DataFrame with survival statistics grouped by class, sex, and age group.
    '''
    # Use the global df loaded in at the top of this file
    
    # define age bins and labels
    age_bins = [0, 12, 19, 59, float('inf')]
    age_labels = ['Child', 'Teen', 'Adult', 'Senior']
    # create categorical age groups with explicit categories so groupby can include empty groups
    df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)
    df['age_group'] = df['age_group'].astype(pd.CategoricalDtype(categories=age_labels, ordered=True))

    # ensure pclass and sex are categorical with expected categories so all combinations appear
    # use the lowercase 'pclass' and 'sex' columns which tests expect
    df['pclass'] = df['pclass'].astype(pd.CategoricalDtype(categories=[1, 2, 3], ordered=True))
    # create a lowercase sex alias and set categorical dtype
    df['sex'] = df['Sex'].astype(pd.CategoricalDtype(categories=['male', 'female']))

    # pass observed=False to include category combinations that have no members
    grouped = df.groupby(['pclass', 'sex', 'age_group'], observed=False)
    summary = grouped.agg(
        n_passengers=('PassengerId', 'count'),
        n_survivors=('Survived', 'sum')
    ).reset_index()
    # fill missing groups (resulting from categorical combinations) with zeros
    summary['n_passengers'] = summary['n_passengers'].fillna(0).astype(int)
    # n_survivors might be float after aggregation; fillna and cast to int
    summary['n_survivors'] = summary['n_survivors'].fillna(0).astype(int)

    # safe division: if n_passengers is 0 set survival_rate to 0
    summary['survival_rate'] = np.where(
        summary['n_passengers'] > 0,
        summary['n_survivors'] / summary['n_passengers'],
        0.0
    )
    # sort using lowercase 'sex' column
    summary = summary.sort_values(by=['pclass', 'sex', 'age_group']).reset_index(drop=True)
    return summary

def visualize_demographic():
    '''
    Function to visualize survival demographics based on groupings and answer question posed.

    Parameters: none

    Returns a Plotly Figure object that visualizes survival statistics and answers 
    if adult women in first class have a higher survival rate than any other demographic group.
    '''
    # Load in data
    summary_df = survival_demographics()

    # Highlight adult women in first class
    highlight = (
        (summary_df['pclass'] == 1) &
        (summary_df['sex'] == 'female') &
        (summary_df['age_group'] == 'Adult')
    )
    summary_df['highlight'] = highlight

    fig = px.bar(
        summary_df,
        x='age_group',
        y='survival_rate',
        color='sex',
        facet_col='pclass',
        barmode='group',
        title='Survival Rate by Class, Sex, and Age Group',
        color_discrete_map={'male': 'steelblue', 'female': 'salmon'},
        hover_data=['n_passengers', 'n_survivors']
    )

    fig.update_traces(marker_line_width=1.5, marker_line_color='black')
    fig.update_layout(showlegend=True)

    return fig

def family_groups():
    '''
    Function to explore the relationship between family size, passenger class, and ticket fare.

    Parameters: none

    Returns a DataFrame grouped by family size and passenger class that displays the number of passengers, average fare, minimum fare, and maximum fare.
    '''
    # Use the global df loaded in at the top of this file
    
    # Calculate family size
    df['family_size'] = df['SibSp'] + df['Parch'] + 1

    # ensure pclass is categorical so grouping behavior is consistent
    if 'pclass' in df.columns:
        df['pclass'] = df['pclass'].astype(pd.CategoricalDtype(categories=[1, 2, 3], ordered=True))

    # Group by family size and class; pass observed=False to retain current behavior
    grouped = df.groupby(['family_size', 'pclass'], observed=False)

    # Aggregate statistics
    summary = grouped.agg(
        n_passengers=('PassengerId', 'count'),
        avg_fare=('Fare', 'mean'),
        min_fare=('Fare', 'min'),
        max_fare=('Fare', 'max')
    ).reset_index()

    # Sort for readability
    summary = summary.sort_values(by=['pclass', 'family_size']).reset_index(drop=True)

    return summary

def last_names():
    '''
    Function to extracts the last name of each and returns the count for each last name.
    
    Parameters: none

    Returns a DataFrame with last names and their counts.
    '''
    # Use the global df loaded in at the top of this file

    # Extract last names
    df['last_name'] = df['Name'].str.extract(r'^([^,]+),')[0]

    # Count occurrences
    last_name_counts = df['last_name'].value_counts()

    return last_name_counts

def visualize_families():
    '''
    Function to visualize family size based on groupings and answer question posed.

    Parameters: none

    Returns a Plotly visualization that answers the question if larger families in third class tend 
    to pay lower fares per person compared to smaller families in first class?
    '''
    # Use the global df loaded in at the top of this file
    summary_df = family_groups()

    # Create scatter plot
    fig = px.scatter(
        summary_df,
        x='family_size',
        y='avg_fare',
    color='pclass',
        size='n_passengers',
        title='Average Fare by Family Size and Class',
        labels={'family_size': 'Family Size', 'avg_fare': 'Average Fare'},
        hover_data=['min_fare', 'max_fare']
    )

    fig.update_layout(showlegend=True)
    return fig