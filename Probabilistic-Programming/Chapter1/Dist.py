import scipy.stats as stats
import numpy as np 
import matplotlib.pyplot as plt 

x = np.arange(16)
#Poisson Distributions are mostly used with counting like population count
#Expected value of the Poisson RV is lambda
poi = stats.poisson
#lambda holds different paramter values for Poisson distribution
#the more the lambda, the larger probabilities given to the larger numbers
lambda_ = [1.5, 4.25]
colours = ["#348ABD", "#A60628"]

dist = poi.pmf(x, lambda_[0])
plt.figure(figsize=(12.5,4))
plt.bar(x, dist, color=colours[0], alpha=0.8, edgecolor=colours[0], lw="3",
	label="$\lambda = %.1f$" % lambda_[0])
dist = poi.pmf(x, lambda_[1])
plt.bar(x, dist, color=colours[1], alpha=0.4, edgecolor=colours[1], lw="3",
	label="$\lambda = %.1f$" % lambda_[1])
plt.xticks(x + 0.05, x)
plt.legend()
plt.ylabel("probability of $k$")
plt.xlabel("$k$")
plt.title("Probability mass function of a Poisson random variable; differing \
$\lambda$ values")
#plt.savefig("plots/pmfPoisson.png")
#plt.show()

#plt.clf()

x = np.linspace(0, 4, 100)
#Exponential Distribution is mostly used with continous values like time, temperature etc.
#The expected value of the Exponential RV is 1/lambda

expo = stats.expon
lambda_ = [0.5, 1]
#the more the lambda, the less probability assigned to the larger values of random variable
plt.figure(figsize=(12.5,4))
for l, c in zip(lambda_, colours):
	dist = expo.pdf(x, scale=1. / l)
	plt.plot(x ,dist, lw=3, color=c, label="$\lambda = %.1f$" % l)
	plt.fill_between(x, dist, color=c, alpha=.33)

    
    

plt.legend()
plt.ylabel("PDF at $z$")
plt.xlabel("$z$")
plt.ylim(0, 1.2)
plt.title("Probability density function of an Exponential random variable; differing $\lambda$");
#plt.savefig("plots/pdfExponential.png")

plt.show()

