import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import lstsq

def make_polynoms(n, X):
	"""
	Helper function which returns the first n primitives of the constant
	function, so the first polynoms with their integration multipliers.

	Arguments
	=========
	 - n: The order of the ODE, and here the number of integrations to perform.
	 - X: The times at which the polynoms should be evaluated.
	
	Returns
	=======
	A list of numpy arrays containing the values of the polynoms.
	"""
	#
	# Initializing with the constant function.
	polynom_values = [np.ones_like(X)]
	#
	# Adding polynomial terms.
	for i in range(1, n):
		polynom_values.append(
			#
			# We generate each polynom from the last by multiplying by the time
			# serie.
			np.multiply(polynom_values[-1], X)
			#
			# We need to divide by the integration constant each time.
			/ i
		)
	#
	# We return the values.
	return polynom_values

def make_primitives(n, X, Y):
	"""
	Helper function to generate the primitives of F.

	Arguments
	=========
	 - n: The order of the ODE, here the number of primitives we will need.
	 - X: The times at which the primitives should be evaluated.
	 - Y: The known values of the function.

	Returns
	=======
	A list of numpy arrays containing the values of the primitives.
	"""
	#
	# Initializing with the values of the function.
	primitive_values = [Y]
	#
	# Adding integrals.
	for i in range(n):
		primitive_values.append(cumulative_trapezoid(
			primitive_values[-1], X, initial=0
		))
	#
	# We return the values, dropping the first one since it was the function
	# itself.
	return primitive_values[1:]

def project_function(n, Y, polynoms, primitives):
	"""
	Helper routine which projects the function onto the space created by its
	primitives and polynoms to find the coefficients we will use to create the
	ODE.

	Arguments
	=========
	 - n: The order of the ODE.
	 - Y: The known values of the function.
	 - polynoms: The values of the polynoms we need, see make_polynoms.
	 - primitives: The values of the primitives we need, see make_primitives.

	Returns
	=======
	The coefficients of the projection of the function onto its primitives and
	the polynoms. Used later to guess the ODE. Note that we return the full
	output from scipy.linear.lstsq, see its documentation for more details. The
	coefficients are the first value of the tuple.
	"""
	#
	# We add the primitives in reverse order since it will make the matrix
	# simpler down the line.
	M = np.transpose(np.vstack(polynoms + list(reversed(primitives))))
	#
	# Solving the linear system using least squares with linear algebra. The
	# minus sign is because compared to the differential equation we want to
	# isolate the last derivate on the other side of the equation.
	return lstsq(M, -Y)

def solve_initial(n, polynoms, projection):
	"""
	Function used to solve the linear system which gives the initial conditions
	for the derivatives of the ODE from the polynom coefficients.

	Arguments
	=========
	 - n: The order of the ODE.
	 - polynoms: The values of the polynoms we need, see make_polynoms. Only the
	 initial values will be used here.
	 - projection: The optimized coefficients for the function, as provided by
	 project_function.
	
	Returns
	=======
	The initial values of the derivatives, which will be used as initial
	conditions for the approximated ODE.
	"""
	#
	# The coefficients for the polynoms.
	pcoefs = projection[0][:n]
	#
	# The coefficients for the primitives in the homogeneous equation.
	hcoefs = projection[0][n:]
	#
	# The values of the polynoms at the initial value.
	P0 = np.array([P[0] for P in polynoms])
	#
	# Matrix of the remaining hcoefs at initial time.
	A = np.array([
		[
			hcoefs[n - 1 - (i - j - 1)] if j < i else int(i==j)
			for j in range(n)
		]
		for i in range(n)
	])
	#
	# Matrix of the polynomial coefficients still present at initial time.
	B = np.array([
		[
			pcoefs[i+j] if i + j < n else 0
			for j in range(n)
		]
		for i in range(n)
	])
	#
	B0 = -np.matmul(B, P0)
	#
	# We solve A * ? = B * P0 to get the initial values of the derivatives
	# for the function. The minus sign is because we are moving the polynoms to
	# the right-hand side for this resolution.
	return np.linalg.solve(A, -np.matmul(B, P0))

def rhodes(n, X, Y):
	"""
	Reverse Homogeneous Ordinary Differential Equation Solver
	=========================================================
	Given a time series, will try to guess the best coefficients for an ODE that
	the data seems to follow.

	Arguments
	=========
	 - n: The order of the ODE we are trying to guess.
	 - X: The time values at which the function is known.
	 - Y: The values of the functions over X.
	
	Returns
	=======
	The list of coefficients of the ODE, the first n+1 coefficients are for the
	ODE and the next n are the initial values of the derivatives.
	"""
	#
	# We build the polynom and primitive values.
	polynoms = make_polynoms(n, X)
	primitives = make_primitives(n, X, Y)
	#
	# We project the function onto the subspace generated by its primitives and
	# the integration polynoms.
	projection = project_function(n, Y, polynoms, primitives)
	#
	# We deduce the initial values of the derivatives from the projection.
	inital_conditions = solve_initial(n, polynoms, projection)
	ode_coefficients = projection[0][n:]
	#
	# We return the final list of coefficients as two np arrays. Both are
	# flipped so that the highest order values comes first, making them easier
	# to use.
	return (
		#
		# We add the implicit '1' for the highest derivative to make the
		# function easier to use. We also reverse the coefficients so that the
		# highest derivative comes first, which the user might expect.
		np.flip(np.append(ode_coefficients, np.ones(1))),
		np.flip(inital_conditions)
	)

def pretty_format(n, ode_coefficients, initial_conditions):
	"""
	Small helper to pretty print an ODE returned by rhodes.
	"""
	#
	# Helper to print derivatives.
	apostrophe = lambda k: "'" * k
	rtext = [
		f"{coef} x f{apostrophe(n-i)}"
		for i, coef in enumerate(ode_coefficients)
	]
	itext = [
		f"f{apostrophe(n-i-1)}(x0) = {init}"
		for i, init in enumerate(initial_conditions)
	]
	return " + ".join(rtext) + " ~= 0\n" + "\n".join(itext)

