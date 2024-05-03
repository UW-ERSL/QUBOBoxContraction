from dimod.reference.samplers import ExactSolver
from pyqubo import Array,Placeholder
from dwave.system import LeapHybridSampler


import numpy as np
import neal

class QUBOBoxSolverClass:
	def __init__(self,beta=0.5,LBox0 = 1,tol = 1e-6,
			  samplingMethod = "simulatedAnnealing", 
			  nSamples = 50,boxMaxIteration = 100):	
		
		self.beta = beta
		self.LBox0 = LBox0
		self.boxMaxIteration = boxMaxIteration
		self.relativeTolerance = tol
		self.samplingMethod = samplingMethod #  exact, simulatedAnnealing, quantumAnnealing
		self.nSamples = nSamples # not relevant for symbolic sampling method
		if (self.samplingMethod == "exact"):
			self.sampler = ExactSolver()
		elif (self.samplingMethod == "simulatedAnnealing"):
			self.sampler = neal.SimulatedAnnealingSampler(); #SimulatedAnnealingSampler()
		elif (self.samplingMethod == "quantumAnnealing"):
			self.sampler = LeapHybridSampler()       
		else:
			print("Invalid sampling method")

		self.nQubitsPerDimension = 2 # don't change
		
	def modelWithPlaceHolders(self):
		# When using PyQUBO, creating a self.model with place holers avoids repeated compiling
		q = self.matrixSize*[None]
		for i in range(self.matrixSize):
			q[i] = Array.create("q[" + str(i)+"]",shape = self.nQubitsPerDimension,vartype = "BINARY")
		c = self.matrixSize*[0]#placeholders
		b = self.matrixSize*[0]#placeholders
		A = self.matrixSize*[0]#placeholders
		x = self.matrixSize*[0]# symbolic via qubits  
		for i in range(self.matrixSize):
			A[i] = self.matrixSize*[0]#placeholders		  
		L = Placeholder('L')
		for i in range(self.matrixSize):
			c[i] = Placeholder('c[%d]' %i)
			b[i] = Placeholder('b[%d]' %i)
			for j in range(self.matrixSize):
				A[i][j] = Placeholder("A[{i}][{j}]".format(i = i, j = j))      
		for i in range(self.matrixSize):
			x[i] = c[i] + L*(-2*q[i][0] + q[i][1])
		   
		H = 0
		for  i in range(self.matrixSize):
			Ax = 0
			for j in range(self.matrixSize):
				Ax = Ax + A[i][j]*x[j]    
			H = H + x[i]*(0.5*Ax) - x[i]*b[i]
		self.model = H.compile()
		return self.model
	
	def QUBOBoxSolve(self,A, b,xGuess = [],debug = False):
		self.matrixSize = A.shape[0]
		self.model = self.modelWithPlaceHolders()
		qSol = self.matrixSize*[None]
		for i in range(self.matrixSize):
			qSol[i] = self.nQubitsPerDimension*[0]
		if (len(xGuess) == 0 ):
			center = self.matrixSize*[0]#numerical
		else:
			center = xGuess
		self.modelDictionary = {}
		for  i in range(self.matrixSize):
			self.modelDictionary['b[%d]' %i] = b[i]
			for j in range(self.matrixSize):
				self.modelDictionary["A[{i}][{j}]".format(i = i, j = j)] = A[i,j]  
		L = self.LBox0
		boxSuccess = True
		nTranslations = 0
		nContractions = 0
		PEHat = 0
		for iteration in range(self.boxMaxIteration):
			#print('Boxiteration:', iteration, L/self.LBox0)
			if (L/self.LBox0 < self.relativeTolerance):
				break
			if (iteration == self.boxMaxIteration):
				break
			self.modelDictionary['L'] =  L
			for  i in range(self.matrixSize):
				self.modelDictionary['c[%d]' %i] = center[i]    
			bqm = self.model.to_bqm(feed_dict = self.modelDictionary)
			if (self.samplingMethod == "exact"):
				results = self.sampler.sample(bqm)
			elif (self.samplingMethod == "simulatedAnnealing"):
				results = self.sampler.sample(bqm, num_reads=self.nSamples)
			elif (self.samplingMethod == "openjijAnnealing"):
				results = self.sampler.sample(bqm, num_reads=self.nSamples)
			elif (self.samplingMethod == "quantumAnnealing"):
				results = self.sampler.sample(bqm)
 
			sample = results.first.sample
			PEStar = results.first.energy 
				
			if (PEStar < PEHat*(1+1e-8)):# Center has moved
				for i in range(self.matrixSize):		
					qSol[i][0]= sample["q["+str(i)+"][0]"]
					qSol[i][1]= sample["q["+str(i)+"][1]"]
				PEHat = PEStar
				for i in range(self.matrixSize):  
					center[i] = center[i] + L*(-2*qSol[i][0] + qSol[i][1])			  
				nTranslations = nTranslations + 1
			else:# Contraction only if we don't translate
				L = L*self.beta
				nContractions = nContractions + 1
			if(debug):
				print('Iter: ' + str(iteration)  + '; center: ' + str(center) + '; PE: ' + str(PEStar) + '; L: ' + str(L))
		if ( L/self.LBox0  > self.relativeTolerance):
			print("Box method did not converge to desired tolerance")
			boxSuccess = False
	
		return [np.array(center),L,iteration,boxSuccess,nTranslations,nContractions,results]
