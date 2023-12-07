# From Example 6.3 in the textbook

# Probability model
#      Observation     Action            Resulting State
P = {('forecast rain', 'bring umbrella', 'rain with umbrella'   ): 0.9,
     ('forecast rain', 'leave umbrella', 'rain without umbrella'): 0.9,
     ('forecast rain', 'bring umbrella', 'sun with umbrella'    ): 0.1,
     ('forecast rain', 'leave umbrella', 'sun without umbrella' ): 0.1,
     ('forecast sun',  'bring umbrella', 'rain with umbrella'   ): 0.2,
     ('forecast sun',  'leave umbrella', 'rain without umbrella'): 0.2,
     ('forecast sun',  'bring umbrella', 'sun with umbrella'    ): 0.8,
     ('forecast sun',  'leave umbrella', 'sun without umbrella' ): 0.8}

# Utilities of each resulting state
U = {'rain with umbrella':    -0.1,
     'rain without umbrella': -1.0,
     'sun with umbrella':      0.9,
     'sun without umbrella':   1.0}
