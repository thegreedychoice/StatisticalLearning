import scipy.stats as stats
import numpy as np 
import matplotlib.pyplot as plt 

"""
This program uses Numpy to plot artificial datsets for the text message example from Chapter 1
"""


def plot_artificial_dataset():
	"""
	We need 3 paramters to define the dataset.
	tau = the day after which the rate at which text message was received change
	(tau marks some significant event during the lifetime of the recorded dataset)
	lambda_1 = the paramter, for the Poisson Distribution representing the count, before tau
	lambda_2 = the paramter for the distribution after tau
	"""
	tau = stats.randint.rvs(0,80)
	alpha = 1./20.   #Assuming 20 is the mean of the count data
	lambda_1, lambda_2 = stats.expon.rvs(scale=1/alpha, size=2)
	data = np.r_[stats.poisson.rvs(mu=lambda_1, size=tau), stats.poisson.rvs(mu=lambda_2, size=80-tau)]
	plt.bar(np.arange(80), data, color="#348ABD")
	plt.bar(tau-1, data[tau-1], color="r", label="user behavior changed")
	plt.xlim(0, 80)


plt.figure(figsize=(12.5, 5))
plt.title("More example of artifical datasets")
for i in range(4):
	plt.subplot(4, 1, i+1)
	plot_artificial_dataset()

plt.show()

