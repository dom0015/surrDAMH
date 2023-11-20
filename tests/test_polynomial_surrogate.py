from surrDAMH.modules.surrogate_poly import PolynomialTrainer
import numpy as np
import time

# binomial - number of terms:
#           degree: 0 1 2  3  4  5  6  7  8  9  
#   of that degree: 1 2 3  4  5  6  7  8  9 10
# all up to degree: 1 3 6 10 15 21 28 36 45 55

observation_operator = lambda x, y: x*x*x*y - y*y
no_snapshots = 20000
no_test_points = 100

np.random.seed(5)
parameters = np.random.rand(no_snapshots,2)
x = parameters[:,0]
y = parameters[:,1]
observations = observation_operator(x,y)

a = np.random.rand(2)
b = np.random.rand(5)
w = np.random.rand(1)

trainer = PolynomialTrainer(2,1,5,"pinv")
trainer.add_data(parameters, observations)
t=time.time()
evaluator = trainer.get_evaluator()
t_elapsed = time.time() - t

test_data = 10*np.random.rand(no_test_points,2)

evaluations = evaluator(test_data)

x_test = test_data[:,0]
y_test = test_data[:,1]
z_test = observation_operator(x_test,y_test)
z_test = z_test.reshape((-1,1))

print(z_test)
print(evaluations)
print(np.sort(np.abs((z_test-evaluations)/z_test),axis=0))
print("time elapsed:", t_elapsed)