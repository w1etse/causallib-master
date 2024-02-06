from sklearn.linear_model import LogisticRegression
from causallib.estimation import IPW 
from causallib.datasets import load_nhefs

data = load_nhefs()
ipw = IPW(LogisticRegression(max_iter=1000))
ipw.fit(data.X, data.a)
potential_outcomes = ipw.estimate_population_outcome(data.X, data.a, data.y)
effect = ipw.estimate_effect(potential_outcomes[1], potential_outcomes[0])