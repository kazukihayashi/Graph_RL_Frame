
import openseespy.opensees as ops
import numpy as np
import Plotter
import os
import time

def LinearAnalysis(nx,ny,node,connectivity,A,I,load,E=2.05e11):
	'''
	nx (number of spans) [1/1]
	ny (number of stories) [1/1]
	node (nodal coordinates) [m]
	A (cross-sectional area) [m^2]
	I (second moment of inertia) [m^4]
	load (nodal loads) [N]
	E (initial elastic modulus) [N/m^2]
	'''

	nk = np.shape(node)[0]
	n_column = (nx+1)*ny
	nm = np.shape(connectivity)[0]

	ops.wipe()
	ops.model('basic','-ndm',2) # ndm: number of dimensions (1,2,3); ndf: number of DOFs (optional)

	# node
	[ops.node(i,float(node[i,0]),float(node[i,1])) for i in range(nk)]

	# boundary condition
	[ops.fix(i,1,1,1) for i in range(nx+1)] # base nodes (OpenSees index starts from 1)

	# Transf
	ColTransfTag=111
	BeamTransfTag=112
	ops.geomTransf('Linear',ColTransfTag) # 'Linear' 'PDelta' 'Corotational'
	ops.geomTransf('Linear',BeamTransfTag) # 'Linear' 'PDelta' 'Corotational'

	# element
	[ops.element('elasticBeamColumn',i,int(connectivity[i,0]),int(connectivity[i,1]),float(A[i]),E,float(I[i]),ColTransfTag) for i in range(n_column)] # (OpenSees index starts from 1)
	[ops.element('elasticBeamColumn',i,int(connectivity[i,0]),int(connectivity[i,1]),float(A[i]),E,float(I[i]),BeamTransfTag) for i in range(n_column,nm)] # (OpenSees index starts from 1)

	# rigid floor assumption
	for i in range(ny):
		master_node = (nx+1)+(2*nx+1)*i # leftmost node is the master node (OpenSees index starts from 1)
		[ops.equalDOF(master_node,master_node+j+1,1) for j in range(2*nx)] # equal deformations in 1(=x) direction

	# time-dependent load coefficient
	ops.timeSeries('Linear',99)
	ops.pattern('Plain',1111,99)

	[ops.load(i,float(load[i,0]),float(load[i,1]),0.0) for i in range(nk)]  # (OpenSees index starts from 1)

	ops.constraints('Plain') # A plain constraint handler can only enforce fix command and equalDOF command. 
	ops.numberer('RCM') # RCM degree-of-freedom numbering object to provide the mapping between the degrees-of-freedom at the nodes and the equation numbers
	ops.system('ProfileSPD') # This is sometimes also referred to as a skyline storage scheme.

	ops.integrator('LoadControl',1.0)
	ops.algorithm('Linear')
	ops.analysis('Static')
	ops.analyze(1)

	disp = np.array([ops.nodeDisp(i) for i in range(nk)])
	force = np.array([ops.basicForce(i) for i in range(nm)]) # 0: axial force 1: moment at i end 2: moment at j end


	return disp,force

def NonlinearAnalysis(nx,ny,node,connectivity,A,I,Zp,Sy,H,load,E=2.05e11,alpha=0.01):
	'''
	nx (number of spans) [1/1]
	ny (number of stories) [1/1]
	node (nodal coordinates) [m]
	A (cross-sectional area) [m^2]
	I (second moment of inertia) [m^4]
	Zp (Full plastic section modulus) [m^3]
	Sy (Yield stress) [N/m^2]
	H (height of the section) [m]
	load (nodal loads) [N]
	E (initial elastic modulus) [N/m^2]
	alpha (tangent after yielding/initial elastic modulus) [1/1]
	'''

	nk = np.shape(node)[0]
	n_column = (nx+1)*ny
	nm = np.shape(connectivity)[0]

	ops.wipe()
	ops.model('basic','-ndm',2) # ndm: number of dimensions (1,2,3); ndf: number of DOFs (optional)

	# node
	[ops.node(i,float(node[i,0]),float(node[i,1])) for i in range(nk)]

	# boundary condition
	[ops.fix(i,1,1,1) for i in range(nx+1)] # base nodes (OpenSees index starts from 1)

	# Transf
	ColTransfTag=111
	BeamTransfTag=112
	ops.geomTransf('PDelta',ColTransfTag) # 'Linear' 'PDelta' 'Corotational'
	ops.geomTransf('Linear',BeamTransfTag) # 'Linear' 'PDelta' 'Corotational'

	### element ###
	Fy = A*Sy # yield force [N] !! not stress [N/m2]
	My = Zp*Sy # yield moment [N･m] !! not stress [N/m2]
	lp = H*0.5 # hinge length at both end [m]

	for i in range(nm):
		ops.section('Elastic',30000+i,float(E),float(A[i]),float(I[i])) # material linear section
		ops.uniaxialMaterial('Steel01',40000+i,float(Fy[i]),float(E*A[i]),alpha) # strain[1/1]-force[N] relationship
		ops.uniaxialMaterial('Steel01',50000+i,float(My[i]),float(E*I[i]),alpha) # curvature[1/m]-moment[N･m] relationship
		ops.section('Aggregator',60000+i,*[40000+i,'P',50000+i,'Mz']) # material nonlinear section
		ops.beamIntegration('HingeRadauTwo',20000+i,60000+i,float(lp[i]),60000+i,float(lp[i]),30000+i) # 'HingeRadau' 'HingeRadauTwo' 'HingeEndpoint' 'HingeMidpoint' 
		if i < n_column:
			ops.element('forceBeamColumn',10000+i,*connectivity[i].tolist(),ColTransfTag,20000+i)
		else:
			ops.element('forceBeamColumn',10000+i,*connectivity[i].tolist(),BeamTransfTag,20000+i)

	# rigid floor assumption
	for i in range(ny):
		master_node = (nx+1)+(2*nx+1)*i # leftmost node is the master node (OpenSees index starts from 1)
		[ops.equalDOF(master_node,master_node+j+1,1) for j in range(2*nx)] # equal deformations in 1(=x) direction

	# time-dependent load coefficient
	ops.timeSeries('Linear',99)
	ops.pattern('Plain',1111,99)

	[ops.load(i,float(load[i,0]),float(load[i,1]),0.0) for i in range(nk)]  # (OpenSees index starts from 1)

	ops.constraints('Plain') # A plain constraint handler can only enforce fix command and equalDOF command. 
	ops.numberer('RCM') # RCM degree-of-freedom numbering object to provide the mapping between the degrees-of-freedom at the nodes and the equation numbers
	ops.system('ProfileSPD') # This is sometimes also referred to as a skyline storage scheme.

	ops.integrator('LoadControl',1.0)
	ops.algorithm('Newton')
	ops.analysis('Static')
	ops.analyze(1)

	disp = np.array([ops.nodeDisp(i) for i in range(nk)])
	force = np.array([ops.basicForce(10000+i) for i in range(nm)]) # 0: axial force 1: moment at i end 2: moment at j end
	hinge = np.abs(force[:,1:]) > np.tile(My,(2,1)).T

	return disp,force,hinge

# nx = 1
# ny = 1
# node = np.array([[0,0],[10,0],[0,4],[5,4],[10,4]],dtype=float)
# connectivity = np.array([[0,2],[1,4],[2,3],[3,4]],dtype=int)
# A = np.ones(connectivity.shape[0])
# I = np.ones(connectivity.shape[0])
# load = np.array([[0,0],[0,0],[0,-1],[0,0],[0,-1]])

# t1 = time.time()
# for i in range(1000):
# 	d,f = LinearAnalysis(nx,ny,node,connectivity,A,I,load)
# t2 = time.time()
# print(t2-t1)
# print(d)
# print(f)

# import plotter
# plotter.Draw(node+d[:,:2]*1e10,connectivity,np.ones(connectivity.shape[0])*200)