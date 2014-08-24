from PyQt4 import QtCore,QtGui

rendering = False

import pyqtgraph as pg

import matplotlib
matplotlib.use('Agg')

import math
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import scipy.optimize
import scipy.ndimage
import matplotlib.pyplot as plt
cmap = plt.get_cmap('bwr')

import socket
mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mysock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);
mysock.bind(('', 1234))
mysock.listen(1)
myconn,myaddr = mysock.accept()

targetsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
targetsocket.connect(('aspire.local', 12345))

import struct
#NODE_COUNT = 16
#NODES_TO_USE = 12
NODE_COUNT = 9
NODES_TO_USE = 9

import weights
class RadioData:
	def initWeights(self):
		self.weights = weights.createWeights(self.nodes, self.resX, self.resY, self._lambda)
		self.iweights = np.linalg.pinv(self.weights, rcond=0.2)
		self.weights = sp.sparse.csc_matrix(self.weights)
		self.weightsT = sp.sparse.csc_matrix(self.weights.transpose())
		#self.iweights = sp.sparse.csc_matrix(np.linalg.pinv(self.weights, rcond=0.2))

		self.guess = np.zeros(self.weights.shape[1])
		diffm1 = np.zeros((self.guess.shape[0], self.guess.shape[0]))
		diffm2 = np.zeros((self.guess.shape[0], self.guess.shape[0]))
		diffm3 = np.zeros((self.guess.shape[0], self.guess.shape[0]))
		diffm4 = np.zeros((self.guess.shape[0], self.guess.shape[0]))
		for xi in range(self.resX):
			for yi in range(self.resY):
				ind = yi*self.resX + xi
				if xi > 0:
					# left
					diffm1[ind][yi*self.resX + xi-1] = -1
					diffm1[ind][yi*self.resX + xi] = 1
				if xi < self.resX-1:
					# right
					diffm2[ind][yi*self.resX + xi+1] = -1
					diffm2[ind][yi*self.resX + xi] = 1
				if yi > 0:
					# up
					diffm3[ind][(yi-1)*self.resX + xi] = -1
					diffm3[ind][yi*self.resX + xi] = 1
				if yi < self.resY-1:
					# down
					diffm4[ind][(yi+1)*self.resX + xi] = -1
					diffm4[ind][yi*self.resX + xi] = 1
		self.diffm1 = sp.sparse.csc_matrix(diffm1)
		self.diffm2 = sp.sparse.csc_matrix(diffm2)
		self.diffm3 = sp.sparse.csc_matrix(diffm3)
		self.diffm4 = sp.sparse.csc_matrix(diffm4)

		return
		#tmpnX = self.weights.shape[0]
		#tmpnY = self.weights.shape[1]

		self.diffm = scipy.ndimage.filters.convolve(np.identity(self.resX), np.array([[0,1,0],[1,-4,1],[0,1,0]]))/2
		self.diffm = self.diffm.reshape((1,-1))

		self.diffm = self.diffm.repeat(self.weights.shape[0], axis=0)
		self.diffm = scipy.sparse.csc_matrix(self.diffm)
		#self.newmat = diffm.repeat(tmpnX, axis=0)
		#self.inA = np.vstack((self.data.weights, self.newmat))

		self.weights = np.matrix(self.weights)
		self.invmat = self.weights.transpose() * self.weights
		self.invmat = self.invmat + 2*(self.diffm.transpose() * self.diffm)
		self.invmat = np.linalg.inv(self.invmat) * self.weights.transpose()
		self.iweights = self.invmat

class FileData(RadioData):
	def __init__(self):
		self.nodes = [
[0,0],
[0,3],
[0,6],
[0,9],
[0,12],
[0,15],
[0,18],
[0,21],
[3,21],
[6,21],
[9,21],
[12,21],
[15,21],
[18,21],
[21,21],
[21,18],
[21,15],
[21,12],
[21,9],
[21,6],
[21,3],
[21,0],
[18,0],
[15,0],
[12,0],
[9,0],
[6,0],
[2,0]]

		self.nodes = [
[0,0],
[0,1],
[0,2],
[1,2],
[2,2],
[2,1],
[2,0],
[1,0],

[0,0.5],
[0,1.5],
[0.5,2],
[1.5,2],
[2,0.5],
[2,1.5],
[0.5,0],
[1.5,0]]

		self.nodes = [
[80,290],
[155,290],
[220,250],
[220,150],
[150,110],
[100,110],
[50,110],
[0,170],
[0,240]
]

		self.notnodes = [
[50, 0],
[100, 0],
[150, 0],
[200, 50],
[200, 100],
[200, 150],
[150, 200],
[100, 200],
[50, 200],
[0, 150],
[0, 100],
[0, 50]
]

		# flip the coordinates to make things work out nicely visually :)
		for i in self.nodes:
#			tmp = i[0]
#			i[0] = i[1]
#			i[1] = tmp
			i[0] = -i[0]
#			i[1] = -i[1]
			i[0] = i[0]/10.0
			i[1] = i[1]/10.0

		self.nodes = self.nodes[:NODES_TO_USE]

		# add some noise to avoid duplicate positions
#		for n in range(len(self.nodes)):
#			pos = self.nodes[n]
#			nodelist = self.nodes[:]
#			del nodelist[n]
#			xposes = [i[0] for i in nodelist]
#			yposes = [i[0] for i in nodelist]
#			while pos[0] in xposes:
#				pos[0] = pos[0] + 0.01*((-1)**n)
#			while pos[1] in yposes:
#				pos[1] = pos[1] + 0.01*((-1)**n)

		"""baselines = []
		for i in self.nodes:
			bl = [0]
			for j in self.nodes:
				bl.append(0)
			baselines.append(bl)
		import csv
		f = open("data/empty.csv")
		mcsv = csv.reader(f)
		for i in mcsv:
			nodeid = int(i[0])
			bl = baselines[nodeid]
			bl[0] = bl[0] + 1
			for j in range(len(self.nodes)):
				val = int(i[j+1])
				bl[j+1] = bl[j+1] + val
		for bl in baselines:
			for j in range(len(self.nodes)):
				bl[j+1] = bl[j+1] / float(bl[0])

		f = open("data/m-2.csv")
		mcsv = csv.reader(f)
		self.states = []
		laststate = 0
		vals = []
		for i in mcsv:
			s = []
			t = 0
			for n in i:
				val = int(n)
				# first column is node id, last three columns are timestamp
				if t > 0 and t <= len(self.nodes):
					val = val - baselines[s[0]][t]
					vals.append(val)
				s.append(val)
				t = t + 1
			# ignore the first states (old data)
			if s[0] != 0 and len(self.states) == 0:
				continue
			# pad out missing data
			while s[0] != laststate:
				self.states.append(None)
				laststate = (laststate + 1) % len(self.nodes)
			self.states.append(s)
			laststate = (laststate + 1) % len(self.nodes)

		self.minstate = np.percentile(vals, 5)
		self.maxstate = np.percentile(vals, 95)"""

		self.states = []
		self.laststate = 0

		# TODO: hack: these are used *relative to baselines*
		self.minstate = -10
		self.maxstate = 10

	def calibrate(self, data):
		baselines = []
		for i in self.nodes:
			bl = [0]
			for j in self.nodes:
				bl.append(0)
			baselines.append(bl)
		vals = []
		for i in data:
			nodeid = int(i[0])
			bl = baselines[nodeid]
			for j in range(len(self.nodes)):
				val = int(i[j+1])
				if val == 127:
					if (bl[0] > 0):
						val = float(bl[j+1] / bl[0]) # TODO: hack
					else:
						val = 0 # TODO: hack
				vals.append(val)
				bl[j+1] = bl[j+1] + val
			bl[0] = bl[0] + 1
		for bl in baselines:
			for j in range(len(self.nodes)):
				if bl[0]:
					bl[j+1] = bl[j+1] / float(bl[0])

		self.baselines = baselines

		#self.minstate = np.percentile(vals, 5)
		#self.maxstate = np.percentile(vals, 95)

calibrating = True

class Viewer(QtGui.QWidget):
	def __init__(self, mydata):
		QtGui.QWidget.__init__(self)
		self.data = mydata
		self.stateid = 0

		# 0.005 is too small, 0.02 is too big
		#self.data._lambda = 0.01
		#self.data._lambda = 0.05
		#self.data._lambda = 0.02
		self.data._lambda = 0.08
		#res = len(self.data.nodes)*3
		#res = 50
		#res = 64
		#res = 24
		global res
		res = 32
		self.data.resX = res
		self.data.resY = res

		self.view = QtGui.QGraphicsView(self)
		#self.view.setRenderHint(QtGui.QPainter.Antialiasing)

		self.scene = QtGui.QGraphicsScene()
		self.view.setScene(self.scene)

		if not rendering:
			self.canvas = QtGui.QLabel(self)

		self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
		self.slider.setTickPosition(QtGui.QSlider.TicksBothSides)
		self.slider.setTickInterval(1)
		self.slider.setMinimum(1)
		self.slider.setMaximum(len(self.data.states)/len(self.data.nodes))
		self.slider.valueChanged[int].connect(self.changeState)

		layout = QtGui.QVBoxLayout(self)
		hlayout = QtGui.QHBoxLayout(self)
		hlayout.addWidget(self.view,1)
		layout.addLayout(hlayout)
		layout.addWidget(self.slider)
		if not rendering:
			hlayout.addWidget(self.canvas,1)
		self.setLayout(layout)

		self.nodes = []
		self.links = []
                self.prevPacket = []
		for i in range(len(self.data.nodes)):
			n = self.data.nodes[i]
                        self.prevPacket.append(None)
			links = []
			for j in range(len(self.data.nodes)):
				if j == i:
					links.append(None)
					continue
				if j < i:
					links.append(self.links[j][i])
					continue
				o = self.data.nodes[j]
				li = self.scene.addLine(QtCore.QLineF(n[0], n[1], o[0], o[1]))
				links.append(li)
			self.links.append(links)
		for i in range(len(self.data.nodes)):
			n = self.data.nodes[i]
			si = QtGui.QGraphicsEllipseItem()
			si.nodeId = i
			si.setRect(-0.25, -0.25, 0.5, 0.5)
			si.setPos(n[0], n[1])
			#si.setBrush(QtGui.QBrush(QtGui.QColor("black"), QtCore.Qt.SolidPattern))
			self.scene.addItem(si)
			self.nodes.append(si)

		self.setMinimumSize(800, 600)

		self.data.initWeights()

		QtCore.QTimer.singleShot(0, self.doIdle)
		#self.initScene()
		#self.updateScene()

	def addData(self, spinPacket, calibrating):
		nodeId = spinPacket[1]
		channelId = spinPacket[-1]
                #if channelId == 26:
                #        self.prevPacket[nodeId-1] = spinPacket
		#if channelId != 26 or not self.prevPacket[nodeId-1]:
		#	return False
                spinPacket = list(spinPacket)
                #spinPacket[nodeId:] = self.prevPacket[nodeId-1][nodeId:]
		if calibrating:
			vals = list(spinPacket[1:2+NODE_COUNT])
			vals[0] = vals[0] - 1
			self.data.states.append(vals)
			return False

		updateNeeded = False
		global lastNode, borked
		if len(self.data.states) == 0:
			lastNode = 0
			borked = False
		if nodeId != 1 and len(self.data.states) == 0:
			QtCore.QTimer.singleShot(0, self.doIdle)
			return
		#print nodeId
		while (lastNode % NODE_COUNT) != nodeId-1:
			if lastNode < 12:
				if not borked:
					print "borked: %d" % nodeId
				borked = True
			self.data.states.append(None)
			lastNode = (lastNode + 1) % NODE_COUNT
		lastNode = nodeId
		if nodeId == 1:
			if len(self.data.states) and not borked:
				print "ok"
				updateNeeded = True
				self.stateid = len(self.data.states)/NODE_COUNT - 1
			borked = False
		vals = list(spinPacket[1:2+NODE_COUNT])
		vals[0] = vals[0] - 1 # node id
		for i in range(len(vals)-1):
			if vals[i+1] == 127:
				vals[i+1] = 0 # TODO: hack
			else:
				vals[i+1] = vals[i+1] - self.data.baselines[nodeId-1][i+1]
		#print vals
		self.data.states.append(vals)
		return updateNeeded

	def doIdle(self):
		updateNeeded = False
		while True:
#			d = None
#			while d != "\xDE\xAD":
#				try:
#					d = myconn.recv(1)
#					if d == "\xDE":
#						print "ok?"
#						d = d + myconn.recv(1)
#				except:
#					break
#			if d != "\xDE\xAD":
#				break
#			print "ok!"

			try:
				data = ""
				c = 2 + 1 + NODE_COUNT*2 + 1
				while c != 0:
					newdata = myconn.recv(c)
					myconn.setblocking(1)
					data = data + newdata
					c = c - len(newdata)
			except:
				break
			myconn.setblocking(0)
			try:
				spinPacket = struct.unpack('<Hb' + NODE_COUNT*2*'b' + 'b', data)
			except:
				continue
			print spinPacket
			global calibrating
			updateNeeded = updateNeeded or self.addData(spinPacket, calibrating)
			if calibrating and len(self.data.states) > NODE_COUNT*50:
				calibrating = False
				self.data.calibrate(self.data.states)
				self.data.states = []
				self.stateid = 0
		if updateNeeded:
			#print "!" + str(self.stateid) + "," + str(len(self.data.states))
			#print self.data.states
			self.updateScene()
			self.repaint()
		QtCore.QTimer.singleShot(0, self.doIdle)

	def changeState(self, value):
		self.stateid = value - 1
		self.updateScene()

	def initScene(self):
		count = len(self.data.nodes)
		ourlines = []
		for n in range(count):
			for i in range(count):
				if i <= n:
					continue
				n1 = self.data.nodes[n]
				n2 = self.data.nodes[i]
				p1 = [float(n1[0]), float(n1[1])]
				p2 = [float(n2[0]), float(n2[1])]
				#if p1[0] == p2[0]: continue
				#if p1[1] == p2[1]: continue
				# first always has lowest x dimension
				if p1[0] < p2[0]:
					tmpline = [np.array(p1), np.array(p2), (n,i)]
				else:
					tmpline = [np.array(p2), np.array(p1), (n,i)]
				ourlines.append(tmpline)

	def updateScene(self):
		for links in self.links:
			for li in links:
				if li:
					col = 220
					pen = QtGui.QPen(QtGui.QColor(col, col, col))
					pen.setWidthF(0.05)
					li.setPen(pen)
		count = NODE_COUNT
		use_count = NODES_TO_USE
		ourlines = []
		mlevels = np.zeros(self.data.weights.shape[0])
		seenlevels = [False]*len(mlevels)
		showMap = False
		for n in range(use_count):
			state = self.data.states[self.stateid*count + n]
			if not state:
				continue
			nodeid = state[0]
			links = self.links[nodeid]
			for i in range(use_count):
				if i == nodeid:
					continue
				li = links[i]
				# TODO: !!
				level = state[i+1]
				level = (level-self.data.minstate) / (1.0*self.data.maxstate-self.data.minstate)
				if level < 0.0:
					print state[i+1]
					level = 0.0
				elif level > 1.0:
					print state[i+1]
					level = 1.0

				if level > 0.6:
					level = 1.0 - level
#				if level > 0.5:
#					level = 0.5 - level
#				level = abs(level - 0.3)

				if i < n:
					index = i*use_count + nodeid
					if seenlevels[index]:
						# average both levels
						mlevels[index] = (mlevels[index] + level) / 2.0
						if showMap:
							col = cmap(mlevels[index])
							pen = QtGui.QPen(QtGui.QColor(col[0]*255, col[1]*255, col[2]*255))
							pen.setWidthF(0.05)
							li.setPen(pen)
						continue
				else:
					index = nodeid*use_count + i

				mlevels[index] = level
				seenlevels[index] = True
				if showMap:
					col = cmap(level)
					pen = QtGui.QPen(QtGui.QColor(col[0]*255, col[1]*255, col[2]*255))
					pen.setWidthF(0.05)
					li.setPen(pen)

		#print mlevels

		#solution = np.linalg.lstsq(self.weights, mlevels, rcond=0.15)[0]
		#solution = np.dot(np.linalg.pinv(self.weights, rcond=0.2), mlevels)
		#solution = np.dot(self.iweights, mlevels)
		#solution = sp.sparse.linalg.lsmr(self.weights, mlevels, maxiter=5)[0]
		#solution = self.weights[37]
		#solution = sum(self.weights)

		#inb = np.hstack((mlevels, np.zeros(self.data.weights.shape[0])))
		#solution = self.data.iweights.dot(mlevels)

		#print "solve begin"
		W = self.data.weights
		WT = self.data.weightsT
		y = mlevels
		WTy = WT*y
		WTW = WT*W
		#alpha = 0.4
		#alpha = 0.5
		#alpha = 0.2
		alpha = 1.0
		beta = 0.001
		#beta = 0.0001
		# initial guess: empty space (all zero)
		#guess = np.zeros(W.shape[1])
		#for i in range(5):
		#	for j in range(5):
		#		guess[(j + self.data.resY/2 - 2)*self.data.resX + self.data.resX/2 - 2 + i] = 1.0
		# initial guess: previous frame
		guess = self.data.guess
		diffm1 = self.data.diffm1
		diffm2 = self.data.diffm2
		diffm3 = self.data.diffm3
		diffm4 = self.data.diffm4
		def TVnorm(x):
			img = x.reshape((self.data.resX, self.data.resY))
			#vert = (img[:-1] - img[1:])**2
			vert = (diffm1*x)**2
			vert = np.sqrt(vert + beta)
			#horz = (img[:,:-1] - img[:,1:])**2
			horz = (diffm3*x)**2
			horz = np.sqrt(horz + beta)
			#return sum(sum(vert)) + sum(sum(horz))
			return sum(vert) + sum(horz)
		def TVgradEntry(val1, val2):
			return (val1 - val2)/np.sqrt((val1 - val2)**2 + beta)
		def TVgrad(v):
			#return np.vectorize(lambda v: v/np.sqrt(v**2 + beta))(x)
			return v/np.sqrt(v**2 + beta)
		def TVderiv(x):
			diff1 = TVgrad(diffm1*x)
			diff2 = TVgrad(diffm2*x)
			diff3 = TVgrad(diffm3*x)
			diff4 = TVgrad(diffm4*x)
			out = diff1+diff2+diff3+diff4
		#	out = np.zeros(x.shape[0])
		#	for xi in range(self.data.resX):
		#		for yi in range(self.data.resY):
		#			index = yi*self.data.resX + xi
		#			val1 = x[index]
		#			value = 0
		#			if xi > 0:
		#				value = value + TVgradEntry(val1, x[index-1])
		#			if xi < self.data.resX-1:
		#				value = value + TVgradEntry(val1, x[index+1])
		#			if yi > 0:
		#				value = value + TVgradEntry(val1, x[index-self.data.resX])
		#			if yi < self.data.resY-1:
		#				value = value + TVgradEntry(val1, x[index+self.data.resX])
		#			out[index] = value
			return out
		def fTV(x):
			lsqm = W*x - y
			lsqnorm = 0.5*lsqm.transpose().dot(lsqm)
			out = lsqnorm + alpha*TVnorm(x)
			return out
		def fTVderiv(x):
			lsqderiv = WTW*x - WTy
			ret = lsqderiv + alpha*TVderiv(x)
			return ret
		solution = scipy.optimize.minimize(fTV, guess, method='L-BFGS-B', jac=fTVderiv, options={'maxiter':1})
		#print "***"
		#print solution
		solution = solution.x
		self.data.guess = solution

		#solution = self.data.iweights.dot(mlevels)

		#solution = self.data.weights[300].todense()

		#print scipy.optimize.check_grad(fTV, fTVderiv, solution)

		#print "***"
		#self.diffm = scipy.ndimage.filters.convolve(np.identity(self.data.weights.shape[1]), np.array([[0,1,0],[1,-4,1],[0,1,0]]))/2

		#inA = scipy.sparse.vstack((self.data.weights, 0.5*self.diffm))

		# *** dumb regularization ***
		#inA = scipy.sparse.vstack((self.data.weights.todense(), np.identity(self.data.weights.shape[1])))
		#newb = np.zeros(self.data.weights.shape[1])
		#inb = np.concatenate((mlevels, newb))
		#solution = np.linalg.lstsq(inA.todense(), inb)[0]

		#newb = np.zeros(self.data.weights.shape[1])
		#inb = np.concatenate((mlevels, newb))
		#inA = self.data.weights
		#inb = mlevels
		#solution = np.linalg.lstsq(inA, inb, rcond=0.1)[0]
		#solution = scipy.sparse.linalg.lsmr(inA, inb, maxiter=3)[0]
		#solution = np.array(solution)
		#solution = np.dot(np.linalg.pinv(inA, rcond=0.2), inb)
	
		#solution = np.linalg.lstsq(self.data.weights.todense(), mlevels, rcond=0.1)[0]

		#solution = sum(self.data.weights.todense())

		sol = solution.reshape((self.data.resX, self.data.resY)).transpose()
		levels = [sol.min(), sol.max()]
		print levels
		# remember: TOP one is red
		#levels = [0, 0.05]
		#levels = [-0.05, 0.05]
		#levels = [-0.01, 0.005]
		#levels = [0.04, 0.08]
		levels = [0, 0.02]
		import pyqtgraph.functions
		#lut = np.array([cmap(math.exp(-x/150.0)) for x in range(256)])*255
		lut = np.array([cmap(1.0 - x/256.0) for x in range(256)])*255
		buf,alpha = pg.functions.makeRGBA(sol,levels=levels,lut=lut)
		img = pg.functions.makeQImage(buf,alpha)
		self.img = img
		#self.canvas.setScaledContents(True)
		#self.canvas.setPixmap(QtGui.QPixmap.fromImage(img))

		#img = img.convertToFormat(QtGui.QImage.Format_ARGB32)
		imgdata = img.constBits()
		imgdata.setsize(res*res*4)
		targetsocket.send(imgdata.asstring())

		#filename = "img%d.png" % self.stateid
		#img.save(filename)

	def showEvent(self, e):
		self.resizeEvent(e)

	def resizeEvent(self, e):
		edge = 0.25
		self.view.fitInView(self.scene.itemsBoundingRect().adjusted(-edge, -edge, edge, edge), QtCore.Qt.KeepAspectRatio)

	def closeEvent(self, e):
		pass

if __name__ == "__main__":
	import signal
	signal.signal(signal.SIGINT, signal.SIG_DFL)
	import sys
	app = QtGui.QApplication(sys.argv)
	mydata = FileData()
	window = Viewer(mydata)
	window.show()
	app.exec_()


