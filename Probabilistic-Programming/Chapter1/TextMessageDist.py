import numpy as np 
import matplotlib.pyplot as plt 
import pymc3 as pm
import theano.tensor as tt


#The following code section is to estimate the belief of lambda the parameter
#so that we could model our text messaging data to a distribution

plt.figure(figsize=(13.5, 4.0))
#Each ith row showing text messages received on the ith day
count_data = np.loadtxt('data/txtdata.csv')
n_count_data = len(count_data)
plt.bar(np.arange(n_count_data), count_data, color="#348ABD")
plt.xlabel("Time (days)")
plt.ylabel("Count of text-messages received")
plt.title("Did user habits change over time?")
plt.xlim(0, n_count_data)
#plt.savefig("plots/textmessagedistribution.png")
#plt.show()


#Define a Pymc3 Model with parameters (lambda1, labda2, tau)
#where lambda1 ~ Exponential(alpha)
#	   lambda2 ~ Exponential(alpha)
#	   tau ~ UniformDistribution(0, n_count_data)
# The value of lambda switches to lambda1 before tau and labda2 after
#These are randomly generated values except for alpha which is the inverse of the mean of 
#the text message count data.
#The values lambda1 and lambda2 are stochastic variables and are treated as 
#random number generators at the backend

with pm.Model() as model:
	alpha = 1. /  count_data.mean()
	lambda_1 = pm.Exponential("lambda_1", alpha)
	lambda_2 = pm.Exponential("lambda_2", alpha)
	tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data-1) 

	idx = np.arange(n_count_data) #Index
	#Since lambda_1 and lambda_2 are random, therefore lambda is also random
	lambda_ =  pm.math.switch(tau>=idx, lambda_1, lambda_2)


#the observation variable combines the data, count_data with the proposed data scheme lambda
#through the use of the observed variable
with model:
	observation = pm.Poisson("obs", lambda_, observed=count_data)

with model:
	step = pm.Metropolis()
	trace = pm.sample(10000, tune=5000, step=step)

lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']
tau_samples = trace['tau']

print n_count_data
plt.figure(figsize=(12.5, 10))
#histogram of the samples:

ax = plt.subplot(311)
ax.set_autoscaley_on(False)

plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_1$", color="#A60628", normed=True)
plt.legend(loc="upper left")
plt.title(r"""Posterior distributions of the variables
    $\lambda_1,\;\lambda_2,\;\tau$""")
plt.xlim([15, 30])
plt.xlabel("$\lambda_1$ value")

ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_2$", color="#7A68A6", normed=True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel("$\lambda_2$ value")

plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=n_count_data, alpha=1,
         label=r"posterior of $\tau$",
         color="#467821", weights=w, rwidth=2.)
plt.xticks(np.arange(n_count_data))

plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(count_data)-20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel("probability");
plt.savefig("plots/parametersposteriordistribution.png")

#Now a fundamental question is why do we need all these posterior samples, like What are we trying to acheive?
#A good question is suppose we want to know for day 35, how many text messages did he receive?
#The answer is the expected value of lambda for the day 35, since the expected value of Poisson RV is lambda.
#How do we find it?
#The above code generates the 10000 posterior samples for lambda1, lambda2, and tau
#And for each sample given what the value of tau is we choose the value of lambda_ using switchpoint
#Now inorder to find the value of text messages count for day t (t = 35), we should find the expected value
#of the posterior lambda for each of the 10000 samples.
#For each sample check the value of tau and then compare with t, such that is (t < tau), choose 
#lambda1 else lambda2, and then just average it up for all the samples

N = tau_samples.shape[0]  #10000
expected_texts_per_day = np.zeros(n_count_data)

plt.figure(figsize=(12.5,5))

for day in range(0, n_count_data):
	#ix is the boolean indexes for all samples
	#e.g., [True,False,True,True,False,...]
	ix = day < tau_samples

	expected_texts_per_day[day] = (lambda_1_samples[ix].sum() + lambda_2_samples[~ix].sum()) / N

plt.plot(range(n_count_data), expected_texts_per_day, color="#E24A33", lw="4",
	label="No of expected text messages received")
plt.xlim(0, n_count_data)
plt.xlabel("Day")
plt.ylabel("Expected Number of text messages")
plt.title("Expected # of text messages received")
plt.ylim(0,60)
plt.bar(np.arange(len(count_data)), count_data, color="#348ABD", alpha=0.65, 
	label="observed texts per day")
plt.legend(loc="upper left")
plt.savefig("plots/num_text_messages_per_day.png")
plt.show()







#plt.show()