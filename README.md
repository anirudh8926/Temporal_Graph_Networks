This is the repository containing the code for the extension to TGNs (Emanuele Rossi) that I'm working on.

Brief Overview of Extension:
The extension is concerning the analysis of the message aggregation techniques that can be used to impact model performance.
There's mainly 2 aggregators that have been described in this paper:

1. Last
2. Mean
   with the former only including the latest update to the node state and mean being the mean of all of the node states.

The extension here, is made with a temporally sensitive time decay aggregator, that produces a weighted mean of the states, according to the temporal distance from the current time of each state.
mathematically(very roughly):
new_state = t(s1)*s1 + t(s2)*s2....... / n(s)
This is to observe if the explicit temporal weighting of the states yields any benefit.

The first phase of the research returned a negative result, (temporal weighting hurts my model). This however was not the most accurate result, as the use of the time encoding as described in TGAT is best functional for absolute timestamps; i.e, from t = 0 to now.

The current state of work is to apply a more appropriate encoding that recognizes time deltas (mainly that includes the cyclic nature of time along with it's ability to also scale linearly) that Time2Vector actually works best with. The Time2Vector implementation is currently pending, and the results might change if the encoding was mainly the issue.

FUTURE SCOPE:
modifying the t(s) function to include a learnable parameter, as most datasets logically will differ in sensitivity wrt to time (for ex, a wikipedia dataset might be less sensitive temporally, but a reddit based or a MOOC based dataset would be a lot more sensitive, as people usually look up related courses or reddit threads a lot more frequently.)
