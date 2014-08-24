import numpy as np

# input:
#   positions[n]: array of (x,y) location pairs for n nodes
#   width, height: size of voxels
#   _lambda: parameter for ellipse width
# output:
#   2D n*(width*height) matrix
def createWeights(positions, width, height, _lambda):
	# calculate the grid positions
	# TODO: the boundaries aren't dealt with nicely here (0.5?!) 
	minX = min(i[0] for i in positions)
	maxX = max(i[0] for i in positions)
	minY = min(i[1] for i in positions)
	maxY = max(i[1] for i in positions)
	margin = 0
	x,xstep = np.linspace(minX-margin, maxX+margin, width, retstep=True)
	y,ystep = np.linspace(minY-margin, maxY+margin, height, retstep=True)
	xi, yi = np.meshgrid(x, y)

	nodecount = len(positions)
	linkcount = nodecount*(nodecount-1)

	# first calculate the distances from each voxel position to each node
	nodedistances = np.zeros((nodecount, width*height))
	for i in range(nodecount):
		d = np.sqrt((xi - positions[i][0])**2 + (yi - positions[i][1])**2)
		nodedistances[i] = d.flatten()

	# then create the actual weights matrix
	weights = np.zeros((linkcount, width*height))
	for i in range(nodecount):
		for j in range(i+1,nodecount):
			# distance modifier: 1.0/sqrt(|distance between nodes|)
			d = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
			# (the second part just creates the ellipse shape)
			weight = (1.0/np.sqrt(d))*(nodedistances[i] + nodedistances[j] < d + _lambda)
			weights[i*nodecount + j] = weight

	return weights
