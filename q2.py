import random, time
from math import sqrt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tabulate import tabulate
class Centroid:
	def __init__(self, uuid, dimensions):
		# Unique identifier for the centroid.
		self.uuid = uuid
		# Label representing the digit given to each cluster at the end of the algorithm
		self.label = -1
		self.values = []


	def update_values(self, inputs):
		'''
		Updates the values of the centroid based on the average of its associated inputs.
		Parameters:
			inputs - The list of all input nodes in the system
		'''
		# Initialises new values list with 0 for every value.
		calc_averages = [0 for inp in range(len(inputs[0].values))]
		count = 0
		for inp in inputs:
			if inp.centroid is self:
				count += 1
				# Increments the average values list with the value of the current input node for each dimension
				for index, value in enumerate(inp.values):
					calc_averages[index] += value
		# If there are no input nodes associated, to avoid division by zero averaging
		if count == 0:
			self.values = [0 for i in range(len(calc_averages))]
		else:
			# Divide every running total in the list by the number of inputs associated to find average
			self.values =  list(map(lambda x: x/count, calc_averages))

	def count_inputs(self, inputs):
		'''
		Counts the number of input nodes associated with the cluster.
		Parameters:
			inputs - The list of all input nodes in the system
		Returns:
			count - The number of input nodes associated with the centroid.
		'''
		count = 0
		for inp in inputs:
			if inp.centroid is self:
				count += 1
		return count

	def get_inputs(self, inputs):
		'''
		Returns a list of all input nodes associated with the cluster.
		Parameters:
			inputs - The list of all input nodes in the system
		Returns:
			clustered - A list of all input nodes associated with the centroid
		'''
		clustered = []
		for inp in inputs:
			if inp.centroid is self:
				clustered.append(inp)
		return clustered

class Input:
	def __init__(self, values, centroid, actual):
		self.values = values
		# The currently associated centroid 
		self.centroid = centroid
		# The actual digit, derived from the dataset. Not used during algorithm 
		self.actual = actual

	def update_classification(self, centroids):
		'''
		Updates the centroid associated with the input by finding the closest centroid by euclidian distance.
		Parameters:
			centroids - List of all centroids present in the system.
		'''
		min_centroid = None
		min_distance = 0
		for centroid in centroids:
			sum_distance = 0
			# Finds the euclidian distance between the node and a centroid for each dimension
			for inp_val, cent_val in zip(self.values, centroid.values):
				sum_distance += (inp_val - cent_val) ** 2
			sum_distance = sqrt(sum_distance)
			# Finds and updates the shortest distance. is None check allows for initialisation of centroid
			if min_distance > sum_distance or min_centroid is None:
				min_distance = sum_distance
				min_centroid = centroid
		self.centroid = min_centroid


def update_centroids(centroids, inputs):
	'''
	Update the centroids values for every centroid in the system.
	Parameters:
		centroids - List of all centroids present in the system
		inputs - List of all input nodes present in the system
	Returns:
		True - At least one centroid changed its value
		False - No centroids changed their values
	'''
	flag_changed = False
	for centroid in centroids:
		old_values = centroid.values
		centroid.update_values(inputs)
		if centroid.values != old_values:
			flag_changed = True
	return flag_changed

def update_classifications(centroids, inputs):
	flag_changed = False
	for inp in inputs:
		old_centroid = inp.centroid
		inp.update_classification(centroids)
		if old_centroid is not inp.centroid:
			flag_changed = True
	return flag_changed


def init():
	start_time = time.time()
	random.seed()
	centroids = [Centroid(i, 64) for i in range(0, 10)]
	digits = load_digits()
	# Only takes 70% of the data (calculated manually) for learning data. The rest is reserved for test data
	inputs = [Input(img, random.choice(centroids), actual) for img, actual in zip(digits['data'][:1257], digits['target'][:1257])]
	update_centroids(centroids, inputs)
	flag_first = True
	while True:
		# If nothing has changed, exit the loop
		if (not update_classifications(centroids, inputs) and not update_centroids(centroids, inputs)) and not flag_first:
			break
		flag_first = False

	taken_modes = []
	rows = []
	for centroid in centroids:
		total = 0
		inp_digits = [inp.actual for inp in centroid.get_inputs(inputs)]
		mode_dict = {}
		# Finds modal value for each cluster to assign digit 
		for inp in centroid.get_inputs(inputs):
			if inp.actual in mode_dict:
				mode_dict[inp.actual] += 1
			else:
				mode_dict[inp.actual] = 1
		# Prevents two clusters from having the same label
		for taken_mode in taken_modes:
			mode_dict[taken_mode] = 0

		unique_mode = max(mode_dict, key=lambda x: mode_dict[x])
		if len(inp_digits) == 0:
			raw_mode = "N/A"
		else:
			raw_mode = max(set(inp_digits), key=inp_digits.count)
		centroid.label = unique_mode
		taken_modes.append(unique_mode)
		# Produce row for table below
		rows.append([centroid.uuid, unique_mode, raw_mode, centroid.count_inputs(inputs)])
		total = 0

	time_elapsed = time.time() - start_time

	print("K-means clustering took %.4f seconds.\n" % time_elapsed)
	print("+----------------------------------------------------------------------------------+")
	print(tabulate(rows, headers=['Centroid UUID', 'Unique Digit Label', 'Raw Digit Label', 'No of Input Nodes'], tablefmt='orgtbl'))
	print("+----------------------------------------------------------------------------------+\n")
	print("--------------------------------------TESTING--------------------------------------")
	
	# Takes the remaining inputs for use in testing
	testing_data = [Input(img, None, actual) for img, actual in zip(digits['data'][1257:], digits['target'][1257:])]
	# Makes predictions for which centroid each test item belongs to. Effectively predicting digit value.
	update_classifications(centroids, testing_data)
	percent = 0
	# Runs testing data to calculate the percentage of accuracy
	for test in testing_data:
		if test.centroid.label == test.actual:
			percent += 1
	percent = (percent / len(testing_data)) * 100
	print("Overall accuracy on test data: %.2f%%" % percent)
	reduce_and_plot(centroids, inputs)


def reduce_and_plot(centroids, inputs):
	pca = PCA(n_components=3)
	# Performs PCA algorithm to reduce dimensions to three for graphing
	inputs_transformed = pca.fit_transform([inp.values for inp in inputs])

	centroid_inputs = {}
	# Produces dictionary so different centroid clusters can be coloured differently
	for centroid in centroids:
		centroid_inputs[str(centroid.label)] = {}
		centroid_inputs[str(centroid.label)]['x'] = []
		centroid_inputs[str(centroid.label)]['y'] = []
		centroid_inputs[str(centroid.label)]['z'] = []

	# Populates graphing data with inputs dimensional value
	for inp, inp_transformed in zip(inputs, inputs_transformed):
		centroid_lbl = inp.centroid.label
		centroid_inputs[str(centroid_lbl)]["x"].append(inp_transformed[0])
		centroid_inputs[str(centroid_lbl)]["y"].append(inp_transformed[1])
		centroid_inputs[str(centroid_lbl)]["z"].append(inp_transformed[2])	

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_title("Reduced-dimension Input Nodes Coloured by Cluster")

	# Graph each cluster in a different random colour.
	for key, val in sorted(centroid_inputs.items()):
		ax.scatter(val['x'], val['y'], val['z'], s=30, label="Centroid "+str(key))
	ax.legend()
	plt.show()

if __name__ == "__main__":
	init()
