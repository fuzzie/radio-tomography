from PyQt4 import QtCore,QtGui

import time
import sys
import socket
mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mysock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);
mysock.bind(('', 12345))
mysock.listen(1)
myconn,myaddr = mysock.accept()

class Viewer(QtGui.QWidget):
	def __init__(self):
		QtGui.QWidget.__init__(self)

		self.canvas = QtGui.QLabel(self)
		layout = QtGui.QVBoxLayout(self)
		layout.addWidget(self.canvas)
		self.setLayout(layout)

app = QtGui.QApplication(sys.argv)
window = Viewer()
#window.show()
window.showMaximized()

res = 32

while True:
	c = res*res*4
	data = ""
	while c != 0:
		mydata = myconn.recv(c)
		c = c - len(mydata)
		data = data + mydata
	img = QtGui.QImage(data, res, res, QtGui.QImage.Format_ARGB32)
#	window.canvas.setScaledContents(True)
	window.canvas.setPixmap(QtGui.QPixmap.fromImage(img))
	window.repaint()
	app.processEvents()
	time.sleep(0.01)


