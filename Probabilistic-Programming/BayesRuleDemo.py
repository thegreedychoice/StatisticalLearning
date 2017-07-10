"""
This program showcases a simple case of Bayes' rule as to how the prior Probability of certain event can affect
its posterior probability. 
Problem :
Let A be an event that certain piece of code has no bugs.
Let X be an event all debugging test on the code has passed and it showed no bugs

Find the probability that after gaining new evidence X, how the posterior probability P(A|X) is calculated,
given that we had P(A) as the prior probability representing the probability/belief that the code is bug-free.

P(A|X) = (P(X|A)P(A)) / P(X)
	where P(X) = P(X|A)P(A) + P(X|~A)p(~A)
	P(~A) is the probability that the code has bugs
	P(X|~A) is the probability the debugging tests all passed despite the code having bugs.

For the sake of this example, lets assume that the P(X|~A) = 0.5
p is the prior probability P(A) of event A
Also P(X|A) = 1, since the code has no bugs, so the tests should all pass for certain.

So, P(A|X) = 1.p / (1.p + 0.5(1-p)) = 2p / (1+p)
"""

import matplotlib.pyplot as plt
import numpy as np 


f = plt.figure(1,figsize=(12.5, 4))
p = np.linspace(0,1, 50)
plt.plot(p, 2*p / (1+p), color="#348ABD", lw=3)
plt.fill_between(p, 2*p/(1+p), alpha=.5, facecolor=["#A60628"])
plt.scatter(0.2, 2 * (0.2) / 1.2, c="#348ABD", s=140)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Prior, $P(A) = p$")
plt.ylabel("Posterior, $P(A|X)$, with $P(A) = p$")
plt.title("Is my code bug-free?")
f.show()


colours = ["#348ABD", "#A60628"]
prior = [0.20, 0.80]
posterior = [1./3, 2./3]
g = plt.figure(2,figsize=(12.5, 4))
plt.bar([0.0, 0.7], prior, alpha=0.7, width=0.25, color=colours[0], label="prior distribution",
	lw="3", edgecolor=colours[0])
plt.bar([0.0+0.25, 0.7+0.25], posterior, alpha=0.7, width=0.25, color=colours[1], label="posterior distribution",
	lw="3", edgecolor=colours[1])
plt.ylim(0,1)
plt.xticks([0.20, .95], ["Bugs Absent", "Bugs Present"])
plt.title("Prior and Posterior probability of bugs present")
plt.ylabel("Probability")
plt.legend(loc="upper left");

g.show()

plt.show()

