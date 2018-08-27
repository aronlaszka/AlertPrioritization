import csv
import random
import math
import numpy as np

n_sample = 1000 # number of sampling 
n_ben = 900 # number of benign alerts in each sample
n_mal = 100 # number of attacks in each sample
n_attack = 8
n_alert = 5

data = []
filename = '../data/credit.csv'
with open(filename) as f:
    reader = csv.reader(f)
    data = list(reader)

# 1. Get distributions of background false positive alerts
FP_alert = np.zeros((n_alert, n_sample))
for i in range(n_sample):
	# Get the benigns
	ben = random.sample(data, n_ben)
	for j in range(n_ben):
		if ben[j][0] == 'A14':
			FP_alert[0][i] += 1
		if ben[j][0] == 'A11' and (ben[j][3] == 'A40' or ben[j][3] == 'A46'):
			FP_alert[1][i] += 1
		if (ben[j][0] == 'A12' or ben[j][0] == 'A13') and ben[j][3] == 'A40' and (ben[j][16] == 'A171' or ben[j][16] == 'A172'): 
			FP_alert[2][i] += 1
		if (ben[j][0] == 'A12' or ben[j][0] == 'A13') and (ben[j][3] == 'A42' or ben[j][3] == 'A43' or ben[j][3] == 'A44') and (ben[j][16] == 'A171' or ben[j][16] == 'A172'):
			FP_alert[3][i] += 1
		if (ben[j][0] == 'A12' or ben[j][0] == 'A13') and ben[j][2] == 'A34' and ben[j][3] == 'A49':
			FP_alert[4][i] += 1
print("The mean of each type of alert")
for i in range(n_alert):
	print("Alert type", i+1, ":", math.ceil(sum(FP_alert[i])/n_sample))

# 2. Get probability of triggering an alert for various alert types
mal = random.sample(data, n_mal)
pr_alert = np.zeros((n_attack, n_alert))
attack_type = ['A40','A41','A42','A43','A44','A45','A46','A49']
for i in range(n_attack):
	for j in range(n_mal):
		mal[j][3] = attack_type[i]
		if mal[j][0] == 'A14':
			pr_alert[i][0] += 1
		if mal[j][0] == 'A11' and (mal[j][3] == 'A40' or mal[j][3] == 'A46'):
			pr_alert[i][1] += 1
		if (mal[j][0] == 'A12' or mal[j][0] == 'A13') and mal[j][3] == 'A40' and (mal[j][16] == 'A171' or mal[j][16] == 'A172'): 
			pr_alert[i][2] += 1
		if (mal[j][0] == 'A12' or mal[j][0] == 'A13') and (mal[j][3] == 'A42' or mal[j][3] == 'A43' or mal[j][3] == 'A44') and (mal[j][16] == 'A171' or mal[j][16] == 'A172'):
			pr_alert[i][3] += 1
		if (mal[j][0] == 'A12' or mal[j][0] == 'A13') and mal[j][2] == 'A34' and mal[j][3] == 'A49':
			pr_alert[i][4] += 1
pr_alert /= 100.0
print("Probability of triggering an alert for various alert types")
print(pr_alert)
