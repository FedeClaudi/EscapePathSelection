from scipy.stats import fisher_exact
from rich import print
from myterial import blue_light

def fisher(table, event_name):
    '''
        Fisher exact test for the difference of two binomial distributions
    '''

    _, pval = fisher_exact(table)
    if pval < 0.05:
        print(f'[{blue_light}]The probability of {event_name} is [green]different[/green] between the two conditions with p value: {pval}')
    else:
        print(f'[{blue_light}]The probability of {event_name} is [red]NOT different[/red] between the two conditions with p value: {pval}')