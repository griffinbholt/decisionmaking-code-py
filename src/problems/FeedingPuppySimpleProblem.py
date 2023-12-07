# From Exercises 6.6 and 6.7 in the textbook

# Utilities
U = {('not feed', 'not hungry'): 0.0,
     ('not feed', 'hungry'):    -1.0,
     ('feed', 'not hungry'):    -0.5,
     ('feed', 'hungry'):        -0.1}

F = ['not feed', 'feed'] # Actions
H = ['not hungry', 'hungry'] # Hunger (State)
R = ['not fed recently', 'fed recently'] # Recently Fed (Observation)