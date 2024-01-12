import colorsys
from matplotlib import pyplot
from matplotlib.lines import Line2D
import numpy as np
import os

if not os.path.exists("result"):
	os.makedirs("result")

def Draw(node, connectivity, line_width, node_color=None, line_color=None, node_text=None, line_text=None, vector=None, hinge=None, scale=0.006, name=0, show=False, title=None): # model b: scale = 0.008 model c: scale = 0.006, else 0.01
	"""
	node[nk][3]or[nk][2]:(float) nodal coordinates
	connectivity[nm][2]	:(int)   connectivity to define member
	node_color[nk]		:(str)   color of node
	line_width[nm]		:(float) size of member
	line_color[nm]		:(str)   color of member
	"""
	##if len(connectivity) != len(section) or len(connectivity) != len(stress):
	##	raise ValueError("The size of connectivity, section, and stress must be equal.")
	
	node = np.array(node, dtype = float)
	connectivity = np.array(connectivity, dtype = int)
	line_width = np.array(line_width, dtype = float)
	if line_color is None:
		line_color = ['black' for i in range(len(connectivity))]
	
	fig = pyplot.figure()

	if len(node[0]) == 2:

		ax = pyplot.subplot()
			
		if vector is not None:
			for i in range(len(connectivity)):
				line = Line2D([node[connectivity[i,0],0],node[connectivity[i,1],0]],[node[connectivity[i,0],1],node[connectivity[i,1],1]],linewidth=1,color=line_color[i],zorder=1)
				ax.add_line(line)
			ax.quiver(node[:,0],node[:,1],node[:,0]+vector[:,0],node[:,1],color='red',zorder=2)
			ax.quiver(node[:,0],node[:,1],node[:,0],node[:,1]+vector[:,1],color='red',zorder=2)
		else:
			for i in range(len(connectivity)):
				line = Line2D([node[connectivity[i,0],0],node[connectivity[i,1],0]],[node[connectivity[i,0],1],node[connectivity[i,1],1]],linewidth=line_width[i]*scale,color=line_color[i],zorder=1)
				ax.add_line(line)
				if line_text is not None:
					ax.text((node[connectivity[i,0],0]+node[connectivity[i,1],0])/2,(node[connectivity[i,0],1]+node[connectivity[i,1],1])/2,line_text[i],horizontalalignment='center', verticalalignment='center',fontsize=10,fontweight='bold',color='White',bbox=dict(boxstyle='round,pad=0',fc=line_color[i], ec=line_color[i])) # default: fontsize = 10 (a: 15, b: 12 c: 10)
		
		if node_color is None:
			node_color = ['black' for i in range(len(node))]
		else:
			for i in range(len(node)):
				ax.plot([node[i,0]],[node[i,1]], "o", color=node_color[i], ms=4, zorder=3)
		if node_text is not None:
			for i in range(len(node)):
				ax.text(node[i,0]+0.4,node[i,1]+0.7,node_text[i],fontsize=10, color='Black') # default: fontsize = 8

		if hinge is not None:
			for i in range(connectivity.shape[0]):
				for j in range(2):
					if hinge[i,j]:
						ax.plot([node[connectivity[i,j],0]*0.9+node[connectivity[i,(j+1)%2],0]*0.1],[node[connectivity[i,j],1]*0.9+node[connectivity[i,(j+1)%2],1]*0.1], "o", color='red', ms=2, zorder=4)

		pyplot.xlim([np.min(node[:,0]),np.max(node[:,0])])
		pyplot.ylim([np.min(node[:,1]),np.max(node[:,1])])
		pyplot.tick_params(labelbottom="off",bottom="off",labelleft="off",left="off")
		pyplot.axis('scaled')
		pyplot.axis('off')
		if title is not None:
			pyplot.title(title)
		if type(name) is int:
			# pyplot.savefig(r'result\{0:0=4}.pdf'.format(name),transparent=True)
			pyplot.savefig(r'result/{0:0=4}.pdf'.format(name),dpi=150,transparent=True)
		elif type(name) is str:
			pyplot.savefig(r'result/{0}.pdf'.format(name),transparent=True)
			# pyplot.savefig(r'result\{0}.png'.format(name),dpi=150,transparent=True)
		if show:
			pyplot.show()
		pyplot.close()

	else:
		raise TypeError("node must eliminate z coordinates.")

	return

def graph(y,name=0):
	x = np.linspace(0,len(y)-1,len(y)).astype(int)
	pyplot.figure(figsize=(10,4))
	pyplot.plot(x,y,linewidth=1)
	xt = np.linspace(0,len(y)-1,11).astype(int)
	pyplot.xticks(xt,xt)

	# save figure
	pyplot.savefig(r"result/graph("+str(name)+").png")
	##pyplot.show()
	pyplot.close()

