# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:22:25 2024

@author: ksure
"""
import numpy as np
import QUBOBoxSolver 
import random
import matplotlib.pyplot as plt


plt.close('all')
def generatePDMatrix(d):
	B = np.random.rand(d, d)
	A = -(B + B.transpose())/2
	A = A + d*np.eye(d)
	return A

def createSolutionVector(d):
	return np.array([random.uniform(-2, 1) for _ in range(d)])


expt = 0
if (expt == 0): # random SPD matrices
	d = 2
	nSamples = 20
	maxBoxIterations = 100
	samplingMethod = "simulatedAnnealing"
	nTests = 10
	betaValues = [0.05,0.075, 0.1, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.4, 0.45, 0.5]
	eps_values = [1e-6,1e-8]
elif (expt == 1): # 1D Poisson matrix
	A = np.array([[6,-6,0,0,0,0],[-6,12,-6,0,0,0],[0,-6,12,-6,0,0],[0,0,-6,12,-6,0],
			      [0,0,0,-6,12,-6],[0,0,0,0,-6,12]])
	xExact = np.array([-np.pi/9, np.pi/11, -np.pi/20 ,np.pi/8,  0.05*np.pi, -np.pi/5 ])
	b = A.dot(xExact)
	nSamples = 20
	maxBoxIterations = 100
	samplingMethod = "simulatedAnnealing"
	nTests = 1
	betaValues = [0.05,0.075, 0.1, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.4, 0.45, 0.5]
	#betaValues = [0.195,0.20121,0.205]
	eps_values = [1e-6,1e-8]

elif (expt ==2): # quantum
	A = np.array([[6,-6,0,0,0,0],[-6,12,-6,0,0,0],[0,-6,12,-6,0,0],[0,0,-6,12,-6,0],
			      [0,0,0,-6,12,-6],[0,0,0,0,-6,12]])
	xExact = np.array([-np.pi/9, np.pi/11, -np.pi/20 ,np.pi/8,  0.05*np.pi, -np.pi/5 ])
	b = A.dot(xExact)
	nSamples = 20
	maxBoxIterations = 100
	samplingMethod = "quantumAnnealing"
	nTests = 1
	betaValues = [0.05, 0.2, 0.5]
	eps_values = [1e-6]
	
# Initialize the data for average iterations for each epsilon
avgIters_data = {eps: np.zeros(len(betaValues)) for eps in eps_values}

# Iterate through each beta value and epsilon value
for beta_idx, beta in enumerate(betaValues):
		print('beta: ', beta)
		for  eps  in eps_values:
			total_iterations = []
			# Repeat the solver for each combination 
			for k in range(nTests):
				QB = QUBOBoxSolver.QUBOBoxSolverClass(
					beta=beta,
					LBox0=1,
					tol=eps,
					samplingMethod=samplingMethod,
					nSamples=nSamples,
					boxMaxIteration=maxBoxIterations,
					)
				if (expt == 0):
					A = generatePDMatrix(d)
					xExact = createSolutionVector(d)
					b = A.dot(xExact)

				_, _, nIterations, _, _, _, _ = QB.QUBOBoxSolve(A, b)
				
				total_iterations.append(nIterations)
	
			# Calculate the average iterations for this combination
			avg_iterations = np.mean(total_iterations)

			# Store the average iterations at the correct index
			avgIters_data[eps][beta_idx] = avg_iterations

# Plot the average iterations
plt.figure(figsize=(10, 6))

if (expt == 0):
	plt.plot(betaValues, avgIters_data[eps_values[0]], marker='o', linestyle=':', 
		color='b', label=r'$\epsilon = 10^{-6}$ (SA)')
	plt.plot(betaValues, avgIters_data[eps_values[1]], marker='o', linestyle='-', 
		color='r', label=r'$\epsilon = 10^{-8}$ (SA)')
elif (expt == 1):
	betaQA = [0.05,0.2, 0.5]; 
	NQA = [52, 33,42];# these were obtained by running expt 2
	plt.plot(betaValues, avgIters_data[eps_values[0]], marker='o', linestyle=':', 
		color='b', label=r'$\epsilon = 10^{-6}$ (SA)')
	plt.plot(betaValues, avgIters_data[eps_values[1]], marker='o', linestyle='-', 
		color='r', label=r'$\epsilon = 10^{-8}$ (SA)')
	plt.plot(betaQA, NQA, marker='x', linestyle='--',color='g', label=r'$\epsilon = 10^{-6}$ (QA)')

	# Add title, labels, and legend
plt.xlabel(r'$\beta$', fontsize=18)
plt.ylabel(r'$N$', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()
