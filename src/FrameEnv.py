import numpy as np
np.random.seed(0)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import copy
import Plotter
from numba import f4, i4, b1
import numba as nb
import OpenSees

@nb.njit(parallel=True)
def InitializeGeometry(nx,ny,span,height):
	
	# node
	nk = (nx+1)*(ny+1)
	node = np.zeros((nk,3),dtype=np.float32)
	for i in range(nk):
		iy, ix = np.divmod(i,nx+1)
		node[i,0] = np.sum(span[:ix])
		node[i,1] = np.sum(height[:iy])

	# member
	nm = (2*nx+1)*ny
	n_column = (nx+1)*ny
	connectivity = np.zeros((nm,2),dtype=np.int32)

	count = 0
	# column
	for i in range(ny):
		for j in range(nx+1):
				connectivity[count,0]=i*(nx+1)+j
				connectivity[count,1]=(i+1)*(nx+1)+j
				count += 1
	# beam
	for i in range(ny):
		for j in range(nx):
				connectivity[count,0]=(i+1)*(nx+1)+j
				connectivity[count,1]=(i+1)*(nx+1)+j+1
				count += 1

	member_type = np.zeros(nm,dtype=np.int32) # column = 0, beam = 1
	member_type[n_column:] = 1

	# member lengths
	length = np.zeros(nm,dtype=np.float32)
	for i in range(nm):
		length[i] = np.linalg.norm(node[connectivity[i,1]]-node[connectivity[i,0]])

	return nk,nm,node,connectivity,n_column,member_type,length

@nb.njit(parallel=True)
def InitializeGeometry_dummy(nx,ny,span,height):

	# dummy node
	nk_dummy = (nx+1)*(ny+1) + nx*ny
	node_dummy = np.zeros((nk_dummy,3),dtype=np.float32)
	for i in range(1,nx+1):
		node_dummy[i,0] = np.sum(span[:i])
	for i in range(nx+1,nk_dummy):
		iy, ix = np.divmod(i+nx,2*nx+1)
		node_dummy[i,0] = np.sum(span[:(ix//2)])+span[((ix%(nx*2))//2)]*(ix%2)*0.5
		node_dummy[i,1] = np.sum(height[:iy])

	# dummy member
	nm_dummy = (3*nx+1)*ny

	connectivity_dummy = np.zeros((nm_dummy,2),dtype=np.int32)
	count = 0
	for i in range(ny): # column
		if i == 0: # bottom columns
			for j in range(nx+1): 
				connectivity_dummy[count] = np.array([j,nx+1+j*2])
				count += 1
		else: # other columns
			for j in range(nx+1): 
				connectivity_dummy[count] = np.array([(2*nx+1)*i-nx+j*2,(2*nx+1)*(i+1)-nx+j*2])
				count += 1
	for i in range(ny): # beam
		for j in range(nx*2): # beam of layer i
			connectivity_dummy[count] = np.array([(2*nx+1)*(i+1)-nx+j,(2*nx+1)*(i+1)-nx+j+1])
			count += 1

	# true-dummy relationship
	n_column = (nx+1)*ny
	nm = (2*nx+1)*ny
	true_to_dummy_edge = []
	for i in range(n_column):
		true_to_dummy_edge.append([i])
	for i in range(nm-n_column):
		true_to_dummy_edge.append([n_column+i*2,n_column+i*2+1])

	# member lengths
	length_dummy = np.zeros(nm_dummy,dtype=np.float32)
	for i in range(nm_dummy):
		length_dummy[i] = np.linalg.norm(node_dummy[connectivity_dummy[i,1]]-node_dummy[connectivity_dummy[i,0]])

	return nk_dummy,nm_dummy,node_dummy,connectivity_dummy,true_to_dummy_edge,length_dummy

@nb.njit(parallel=True)
def compute_load(nx,ny,span,height,section_dummy,length_dummy,C0=0.2):

	### Natural Period
	T = np.sum(height)*0.03

	### static earthquake load

	total_weight_for_earthquake = 0.0 #[N]
	alpha_i_for_earthquake = np.zeros(ny,dtype=np.float32) # ratio of upper mass to total mass
	layerwise_weight_for_frame = np.zeros(ny,dtype=np.float32)
	member_volumes_dummy = section_dummy[:,0]*length_dummy
	total_span = np.sum(span)

	for i in range(ny-1,-1,-1):
		volume_layer_i = np.sum(member_volumes_dummy[i*(nx+1):(i+1)*(nx+1)])/2 # bottom columns
		volume_layer_i += np.sum(member_volumes_dummy[(nx+1)*ny+i*(2*nx):(nx+1)*ny+(i+1)*(2*nx)]) # beams
		if i < ny-1:
			volume_layer_i += np.sum(member_volumes_dummy[(i+1)*(nx+1):(i+2)*(nx+1)])/2 # upper columns
		structural_weight_layer_i = volume_layer_i * 77000 #[N/m^3]
		layerwise_weight_for_frame[i] = structural_weight_layer_i + 3400*np.power(total_span,2)/nx  # 2400 live load(for frame, shop) + 1000(floor, isolation etc.)
		total_weight_for_earthquake += structural_weight_layer_i + 2300*np.power(total_span,2)/nx # 1300 live load(for earthquake, shop) + 1000(floor, isolation etc.)
		alpha_i_for_earthquake[i] = total_weight_for_earthquake

	alpha_i_for_earthquake /= total_weight_for_earthquake
	Ai = 1 + (1/np.sqrt(alpha_i_for_earthquake)-alpha_i_for_earthquake)*2*T/(1+3*T)
	qi = C0 * total_weight_for_earthquake * Ai * alpha_i_for_earthquake

	pi = np.copy(qi)
	for i in range(ny-1):
		pi[i] -= pi[i+1]

	### loading condition

	nk_dummy = (nx+1)*(ny+1) + nx*ny
	load = np.zeros((nk_dummy,6),dtype=np.float32)
	total_span = np.sum(span)

	for i in range(nx+1,nk_dummy):
		story = (i-(nx+1))//(2*nx+1)
		lateral = (i-(nx+1))%(2*nx+1)	

		if lateral == 0:
			spanratio = 0.25*span[0]/total_span
		elif lateral == nx*2:
			spanratio = 0.25*span[-1]/total_span
		elif lateral%2 == 0:
			spanratio = 0.25*(span[(lateral//2)-1]+span[lateral//2])/total_span
		else:
			spanratio = 0.5*span[lateral//2]/total_span
		load[i,0] = pi[story]*spanratio
		load[i,1] = -layerwise_weight_for_frame[story]*spanratio

	return load

@nb.njit(parallel=True)
def allowable_stress(nm,nm_dummy,n_column,section,length,E,Fyc,Fyb,term):
	'''
	(input)
	nm<int>       : number of members
	nm_dummy<int> : number of dummy members (members separated at midpoints for beams)
	n_column<int> : number of columns
	section<float,float>  : nm section data
	Fyc           : yield stress of columns
	Fyb           : yield stress of beams
	term<str>     : "short" or "long" depending on loading condition
	(output)
	allowable_stress_dummy<float>: nm_dummy allowable stresses
	'''

	critical_slenderness_ratio_column = np.sqrt(np.power(np.pi,2)*E/(0.6*Fyc))
	critical_slenderness_ratio_beam = np.sqrt(np.power(np.pi,2)*E/(0.6*Fyb))

	allowable_stress_compression_dummy = np.empty((nm_dummy),dtype=np.float32)

	slenderness_ratio = length * 0.65 / np.sqrt(section[:,3]/section[:,0])
	slenderness_ratio_dummy = np.zeros(nm_dummy)
	for i in range(n_column):
		slenderness_ratio_dummy[i] = slenderness_ratio[i]
	for i in range(nm-n_column):
		slenderness_ratio_dummy[n_column+2*i:n_column+2*(i+1)] = slenderness_ratio[n_column+i] 
	a_c = slenderness_ratio_dummy[:n_column]/critical_slenderness_ratio_column
	allowable_stress_compression_dummy[:n_column][a_c<1] = Fyc*(1.0-0.4*np.power(a_c[a_c<1],2))/(3.0/2.0+np.power(a_c[a_c<1],2)*2.0/3.0)
	allowable_stress_compression_dummy[:n_column][a_c>=1] = Fyc*18.0/(65.0*np.power(a_c[a_c>=1],2))
	a_b = slenderness_ratio_dummy[n_column:]/critical_slenderness_ratio_beam
	allowable_stress_compression_dummy[n_column:][a_b<1] = Fyb*(1.0-0.4*np.power(a_b[a_b<1],2))/(3.0/2.0+np.power(a_b[a_b<1],2)*2.0/3.0)
	allowable_stress_compression_dummy[n_column:][a_b>=1] = Fyb*18.0/(65.0*np.power(a_b[a_b>=1],2))

	allowable_stress_tension_dummy = np.array([Fyc for i in range(n_column)]+[Fyb for i in range(n_column,nm_dummy)]) 
	allowable_stress_bending_dummy = np.array([Fyc for i in range(n_column)]+[Fyb for i in range(n_column,nm_dummy)]) 

	if term == 'short':
		return allowable_stress_compression_dummy/1.0, allowable_stress_tension_dummy/1.0, allowable_stress_bending_dummy/1.0
	else: # term == 'long'
		return allowable_stress_compression_dummy/1.5, allowable_stress_tension_dummy/1.5, allowable_stress_bending_dummy/1.5

# @nb.njit(parallel=True)
def compute_cof(nx,ny,nk,n_column,Fyc,Fyb,Zp,stress_dummy):
	'''
	(input)
	stress_dummy<float>[nm_dummy,3] nm_dummy stresses, with an order that columns come first and beams next.
	sec_num<int>: nm section numbering (200,250,...)
	(output)
	cof<float>: column-to-beam strength ratios for all the nodes
	'''
	cof = np.ones(nk)*np.inf

	for ii in range(nx+1,nk-(nx+1)):

		i = ii//(nx+1)-1
		j = ii%(nx+1)

		# bottom column
		axial_force_ratio = np.abs(stress_dummy[ii-(nx+1),0]/Fyc)
		if(axial_force_ratio < 0.5):
			alpha = 1-4*np.power(axial_force_ratio,2)/3
		else:
			alpha = 4*(1-axial_force_ratio)/3
		Mpc_bottom = np.clip(alpha,0.0,None)*Fyc*Zp[ii-(nx+1)] # alpha is clipped to avoid negative COF

		# upper column
		if (i!=(ny-1)):
			axial_force_ratio = np.abs(stress_dummy[ii,0]/Fyc)
			if(axial_force_ratio < 0.5):
				alpha = 1-4*np.power(axial_force_ratio,2)/3
			else:
				alpha = 4*(1-axial_force_ratio)/3
			Mpc_upper = np.clip(alpha,1.0e-5,None)*Fyc*Zp[ii] # alpha is clipped to avoid negative COF
		else:
			Mpc_upper = 0.0

		# left and right beams
		if(j==0):
			Mpb_left = 0.0
			Mpb_right = Fyb*Zp[n_column+nx*i+j]
		elif(j==nx):
			Mpb_left = Fyb*Zp[n_column+nx*i+j-1]
			Mpb_right = 0.0
		else:
			Mpb_left = Fyb*Zp[n_column+nx*i+j-1]
			Mpb_right = Fyb*Zp[n_column+nx*i+j]

		cof[ii] = (Mpc_bottom+Mpc_upper)/(Mpb_left+Mpb_right)

	return cof

@nb.njit(parallel=True)
def check_collapse(nx,ny,n_column,nm,hinge):
	'''
	## output ##
	collapse <bool>: True if total/layer/beam collapse occurs
	total_collapse <bool>: True if total collapse occurs
	layer_collapse <List<int>>: The structure collapses at the layer
	beam_collapse <List<int>>: Hinges occur at the both ends and midpoint of the beam
	'''
	beam_collapse = []
	for i in range(nm-n_column): # beam collapse
		if np.all(hinge[n_column+2*i:n_column+2*(i+1)]):
			beam_collapse.append(n_column+i)
	
	layer_collapse = []
	for i in range(ny): # layer collapse
		if np.all(hinge[(nx+1)*i:(nx+1)*(i+1)]):
			layer_collapse.append(i)
	
	if len(layer_collapse) > 0 or len(beam_collapse) > 0:
		collapse = True
		total_collapse = False
	else:
		total_collapse = True
		for i in range(ny):
			if i == 0:
				for j in range(nx):
					indices = np.array([j,j+1,n_column+2*j,n_column+2*j+1],dtype=np.int32)
					nh = 0
					for k in range(len(indices)):
						nh += np.sum(hinge[indices[k]])
					if nh - np.int32(hinge[indices[2],1]) < 4:
						total_collapse = False
						break
			elif total_collapse:
				for j in range(nx):
					indices = np.array([(nx+1)*i+j,(nx+1)*i+j+1,n_column+2*nx*(i-1)+2*j,n_column+2*nx*(i-1)+2*j+1,n_column+2*nx*i+2*j,n_column+2*nx*i+2*j+1],dtype=np.int32)
					nh = 0
					for k in range(len(indices)):
						nh += np.sum(hinge[indices[k]])
					if nh - np.int32(hinge[indices[2],1]) - np.int32(hinge[indices[4],1]) < 4:
						total_collapse = False
						break
		if total_collapse:
			collapse = True
		else:
			collapse = False

	return collapse, total_collapse, layer_collapse, beam_collapse

class Frame():

	column_section_list ={
		## A, Ix, Iz(strong), Iy(weak), Zz(strong), Zy(weak) [metric]
		200:(85.3/1.0E4, 2*4860/1.0E8, 4860/1.0E8, 4860/1.0E8, 486/1.0E6, 486/1.0E6), # 200x200x12
		250:(109.3/1.0E4, 2*10100/1.0E8, 10100/1.0E8, 10100/1.0E8, 805/1.0E6, 805/1.0E6), # 250x250x12
		300:(173.0/1.0E4, 2*22600/1.0E8, 22600/1.0E8, 22600/1.0E8, 1510/1.0E6, 1510/1.0E6), # 300x300x16
		350:(239.2/1.0E4, 2*42400/1.0E8, 42400/1.0E8, 42400/1.0E8, 2420/1.0E6, 2420/1.0E6), # 350x350x19 cold-rolled
		400:(307.7/1.0E4, 2*69500/1.0E8, 69500/1.0E8, 69500/1.0E8, 3480/1.0E6, 3480/1.0E6), # 400x400x22 cold-pressed
		450:(351.7/1.0E4, 2*103000/1.0E8, 103000/1.0E8, 103000/1.0E8, 4560/1.0E6, 4560/1.0E6), # 450x450x22
		500:(442.8/1.0E4, 2*159000/1.0E8, 159000/1.0E8, 159000/1.0E8, 6360/1.0E6, 6360/1.0E6), # 500x500x25
		550:(492.8/1.0E4, 2*217000/1.0E8, 217000/1.0E8, 217000/1.0E8, 7900/1.0E6, 7900/1.0E6), # 550x550x25
		600:(542.8/1.0E4, 2*288000/1.0E8, 288000/1.0E8, 288000/1.0E8, 9620/1.0E6, 9620/1.0E6), # 600x600x25
		650:(656.3/1.0E4, 2*407000/1.0E8, 407000/1.0E8, 407000/1.0E8, 12500/1.0E6, 12500/1.0E6), # 650x650x28
		700:(712.3/1.0E4, 2*518000/1.0E8, 518000/1.0E8, 518000/1.0E8, 14800/1.0E6, 14800/1.0E6), # 700x700x28
		750:(866.3/1.0E4, 2*717000/1.0E8, 717000/1.0E8, 717000/1.0E8, 19100/1.0E6, 19100/1.0E6), # 750x750x32
		800:(930.3/1.0E4, 2*884000/1.0E8, 884000/1.0E8, 884000/1.0E8, 22100/1.0E6, 22100/1.0E6), # 800x800x32
		850:(994.3/1.0E4, 2*1070000/1.0E8, 1070000/1.0E8, 1070000/1.0E8, 25300/1.0E6, 25300/1.0E6), # 850x850x32
		900:(1177.0/1.0E4, 2*1420000/1.0E8, 1420000/1.0E8, 1420000/1.0E8, 31500/1.0E6, 31500/1.0E6), # 900x900x36
		950:(1249.0/1.0E4, 2*1680000/1.0E8, 1680000/1.0E8, 1680000/1.0E8, 35500/1.0E6, 35500/1.0E6), # 950x950x36
		1000:(1321.0/1.0E4, 2*1990000/1.0E8, 1990000/1.0E8, 1990000/1.0E8, 39700/1.0E6, 39700/1.0E6) # 1000x1000x36
		}
	column_plastic_section_modulus = {200:588/1.0E6, 250:959/1.0E6, 300:1810/1.0E6, 350:2910/1.0E6, 400:4220/1.0E6, 450:5490/1.0E6, 500:7660/1.0E6, 550:9460/1.0E6, 600:11400/1.0E6, 650:14900/1.0E6, 700:17600/1.0E6, 750:22800/1.0E6, 800:26200/1.0E6, 850:29900/1.0E6, 900:37300/1.0E6, 950:42000/1.0E6, 1000:46900/1.0E6}

	beam_section_list ={
		## A, Ix, Iz(strong), Iy(weak), Zz(strong), Zy(weak) [metric]
		200:(38.11/1.0E4, 2*2630/1.0E8, 2630/1.0E8, 507/1.0E8, 271/1.0E6, 67.6/1.0E6), # 194x150x6x9
		250:(55.49/1.0E4, 2*6040/1.0E8, 6040/1.0E8, 984/1.0E8, 495/1.0E6, 112/1.0E6), # 244x175x7x11
		300:(71.05/1.0E4, 2*11100/1.0E8, 11100/1.0E8, 1600/1.0E8, 756/1.0E6, 160/1.0E6), # 294x200x8x12
		350:(99.53/1.0E4, 2*21200/1.0E8, 21200/1.0E8, 3650/1.0E8, 1250/1.0E6, 292/1.0E6), # 340x250x9x14 middle H
		400:(110.0/1.0E4, 2*31600/1.0E8, 31600/1.0E8, 2540/1.0E8, 1580/1.0E6, 254/1.0E6), # 400x200x9x19 super high-slend H (SHH)
		450:(126.0/1.0E4, 2*45900/1.0E8, 45900/1.0E8, 2940/1.0E8, 2040/1.0E6, 294/1.0E6), # 450x200x9x22
		500:(152.5/1.0E4, 2*70700/1.0E8, 70700/1.0E8, 5730/1.0E8, 2830/1.0E6, 459/1.0E6), # 500x250x9x22
		550:(157.0/1.0E4, 2*87300/1.0E8, 87300/1.0E8, 5730/1.0E8, 3180/1.0E6, 459/1.0E6), # 550x250x9x22
		600:(192.5/1.0E4, 2*121000/1.0E8, 121000/1.0E8, 6520/1.0E8, 4040/1.0E6, 522/1.0E6), # 600x250x12x25
		650:(198.5/1.0E4, 2*145000/1.0E8, 145000/1.0E8, 6520/1.0E8, 4460/1.0E6, 522/1.0E6), # 650x250x12x25
		700:(205.8/1.0E4, 2*173000/1.0E8, 173000/1.0E8, 6520/1.0E8, 4940/1.0E6, 522/1.0E6), # 700x250x12x25
		750:(267.9/1.0E4, 2*261000/1.0E8, 261000/1.0E8, 12600/1.0E8, 6970/1.0E6, 841/1.0E6), # 750x300x14x28
		800:(274.9/1.0E4, 2*302000/1.0E8, 302000/1.0E8, 12600/1.0E8, 7560/1.0E6, 841/1.0E6), # 800x300x14x28
		850:(297.8/1.0E4, 2*355000/1.0E8, 355000/1.0E8, 12600/1.0E8, 8350/1.0E6, 842/1.0E6), # 850x300x16x28
		900:(305.8/1.0E4, 2*404000/1.0E8, 404000/1.0E8, 12600/1.0E8, 8990/1.0E6, 842/1.0E6), # 900x300x16x28
		950:(313.8/1.0E4, 2*458000/1.0E8, 458000/1.0E8, 12600/1.0E8, 9640/1.0E6, 842/1.0E6), # 950x300x16x28
		1000:(321.8/1.0E4, 2*515000/1.0E8, 515000/1.0E8, 12600/1.0E8, 10300/1.0E6, 842/1.0E6) # 1000x300x16x28
	}
	beam_plastic_section_modulus = {200:301/1.0E6, 250:550/1.0E6, 300:842/1.0E6, 350:1380/1.0E6, 400:1770/1.0E6, 450:2750/1.0E6, 500:3130/1.0E6, 550:3520/1.0E6, 600:4540/1.0E6, 650:5030/1.0E6, 700:5580/1.0E6, 750:7850/1.0E6, 800:8520/1.0E6, 850:9540/1.0E6, 900:10300/1.0E6, 950:11100/1.0E6, 1000: 11900/1.0E6}

	# material

	DESIGN_STRENGTH_BEAM = 235e6 #[N/m^2], 235e6 for SN400 and BCP235; 325e6 for SN490 and BCP325
	DESIGN_STRENGTH_COLUMN = 235e6 #[N/m^2], 235e6 for SN400 and BCP235; 325e6 for SN490 and BCP325
	E = 2.05e11
	G = 7.9e10

	# design constraint
	allowable_column_disp = 1/200
	allowable_beam_disp = 1/300
	allowable_column_ultimate_disp = 1/100

	def __init__(self,mode):
		self.mode = mode # 'dec' or 'inc'
		self.reset(test=True)

	def reset(self,test=0):

		if test == 0: ## Randomly generate the structural shape if test == 0
			self.NX = np.random.randint(2,6)
			self.NY = np.random.randint(2,11)
			self.span = 5.0+np.random.rand(self.NX)*10.0
			self.height = np.ones(self.NY)*3.0
			self.height[0] += np.random.rand()*0.5
			self.height += np.random.rand()*1.0
		else: ## Generate a prefixed structural shape otherwise
			if test == 1:
				self.NX = 3 # ex.1: 3; ex.2: 5; ex.3: 6
				self.NY = 8 # ex.1: 8; ex.2: 4; ex.3: 12
				self.span = np.array([12,8,8]) # ex.1: np.array([12,8,8]); ex.2: np.array([8,12,12,10,10]); ex.3: np.array([10,10,6,8,8,8])
			elif test == 2:
				self.NX = 5 # ex.1: 3; ex.2: 5; ex.3: 6
				self.NY = 4 # ex.1: 8; ex.2: 4; ex.3: 12
				self.span = np.array([8,12,12,10,10]) # ex.1: np.array([12,8,8]); ex.2: np.array([8,12,12,10,10]); ex.3: np.array([10,10,6,8,8,8])
			elif test == 3:
				self.NX = 6 # ex.1: 3; ex.2: 5; ex.3: 6
				self.NY = 12 # ex.1: 8; ex.2: 4; ex.3: 12
				self.span = np.array([10,10,6,8,8,8]) # ex.1: np.array([12,8,8]); ex.2: np.array([8,12,12,10,10]); ex.3: np.array([10,10,6,8,8,8])
			self.height = np.ones(self.NY)*3.5 # ex.1 2 and 3: 3.5
			self.height[0] = 4.0 # ex.1 2 and 3: 4.03.5


		self.nk,self.nm,self.node,self.connectivity,self.n_column,self.member_type,self.length = InitializeGeometry(self.NX,self.NY,self.span,self.height)
		self.nk_dummy,self.nm_dummy,self.node_dummy,self.connectivity_dummy,self.true_to_dummy_edge,self.length_dummy = InitializeGeometry_dummy(self.NX,self.NY,self.span,self.height)

		self.dependency = dict()
		self.dependency[1] = []
		self.dependency[0] = [] 
		# dependency of members
		for i in range(self.n_column): # column
			self.dependency[1].append(np.arange(i-(self.NX+1),-1,-(self.NX+1)))
			self.dependency[0].append(np.arange(i+(self.NX+1),self.n_column,self.NX+1))
		for i in range(self.n_column,self.nm): # beam
			self.dependency[1].append(np.array([],dtype=int))
			self.dependency[0].append(np.array([],dtype=int))
		
		# initialize RL episode
		self.done = False
		self.steps = 0
		self.total_reward = 0
		self.selected_action = None

		if self.mode == 'dec':
			self.sec_num = np.ones(self.nm,dtype=int)*1000 # !!! reducing size
		elif self.mode == 'inc':
			self.sec_num = np.ones(self.nm,dtype=int)*200 # !!! increasing size

		# initialize edge selection
		self.infeasible_action = np.zeros((self.nm,2),dtype=bool)
		if self.mode == 'dec':
			self.infeasible_action[:,1] = True # !!! reducing size
		elif self.mode == 'inc':
			self.infeasible_action[:,0] = True # !!! increasing size
		
		# initialize node and edge inputs
		self.v,self.w,_,cofs,deform_r_c,deform_r_b,stress_ratio,_,collapse,_,_,self.volume = self.update_state(self.sec_num)
		self.feasible = self.check_feasibility(stress_ratio,cofs,deform_r_c,deform_r_b,collapse)

		self.Max_stress_ratio_before = np.max(stress_ratio)
		self.Max_disp_ratio_before = np.max(np.concatenate([deform_r_c/self.allowable_column_disp,deform_r_b/self.allowable_beam_disp]))
		self.Min_cof_before = np.min(cofs)

		return np.copy(self.v),np.copy(self.w),np.copy(self.connectivity),np.copy(self.infeasible_action)

	def step(self, action):
		'''
		action[int,int]:
		0: member index and 1: action type(0:size down 1: size up)
		'''
		volume_before = np.copy(self.volume)

		# proceed to next step
		self.selected_action = action[0]
		typedict = {0:-50,1:50}
		observation_value = self.sec_num[action[0]] + typedict[action[1]]

		# confirm if satisfy constraints
		self.sec_num[action[0]] = observation_value
		n_change = 1
		for dependent_member in self.dependency[action[1]][action[0]]:
			if action[1] == 0:
				if self.sec_num[dependent_member] > observation_value:
					self.sec_num[dependent_member] = observation_value
					n_change += 1
			elif action[1] == 1:
				if self.sec_num[dependent_member] < observation_value:
					self.sec_num[dependent_member] = observation_value
					n_change += 1
					
		self.infeasible_action[(self.sec_num == 200),0] = True
		self.infeasible_action[(self.sec_num == 1000),1] = True
				
		self.v,self.w,_,cofs,deform_r_c,deform_r_b,stress_ratio,_,collapse,_,_,self.volume = self.update_state(self.sec_num)
		V_diff = self.volume - volume_before # V_diff is positive if increasing size and negative if reducing size
		# print(V_diff)

		# if collapse:
		# 	print('COLLAPSE')
		self.feasible = self.check_feasibility(stress_ratio,cofs,deform_r_c,deform_r_b,collapse)

		Max_stress_ratio = np.max(stress_ratio)
		Max_disp_ratio = np.max(np.concatenate([deform_r_c/self.allowable_column_disp,deform_r_b/self.allowable_beam_disp]))
		Min_cof = np.min(cofs)
		
		if self.mode == 'dec':
			if self.feasible: # !!! reducing size
				r1 = np.clip(self.Max_stress_ratio_before/Max_stress_ratio,0.0,0.99)
				r2 = np.clip(self.Max_disp_ratio_before/Max_disp_ratio,0.0,0.99)
				sv = Max_stress_ratio - np.min(stress_ratio)
				reward = 0.1*np.sqrt(-V_diff)/sv*-(np.log(1-r1) + np.log(1-r2)) #/n_change
			else:
				reward = -1.0
				self.done = True

		elif self.mode == 'inc':
			if self.feasible: # !!! increasing size
				reward = +1.0
				self.done = True
			else:
				r1 = np.clip(Max_stress_ratio/self.Max_stress_ratio_before,None,0.99) if Max_stress_ratio > 1.0 else 0.0
				r2 = np.clip(Max_disp_ratio/self.Max_disp_ratio_before,None,0.99) if Max_disp_ratio > 1.0 else 0.0
				r3 = np.clip(self.Min_cof_before/Min_cof,None,0.99) if Min_cof < 1.0 else 0.0
				sv = Max_stress_ratio - np.min(stress_ratio)
				reward = 0.1*np.sqrt(V_diff)*sv*(np.log(1-r1) + np.log(1-r2) + np.log(1-r3))

		if np.all(self.infeasible_action):
			self.done = True

		self.Max_stress_ratio_before = Max_stress_ratio
		self.Max_disp_ratio_before = Max_disp_ratio
		self.Min_cof_before = Min_cof

		self.total_reward += reward
		self.steps += 1

		return np.copy(self.v),np.copy(self.w), reward, self.done, np.copy(self.infeasible_action)

	def update_state(self,sec_num,v=None,w=None):

		section = np.array([self.column_section_list[sec_num[i]] for i in range(self.n_column)]+[self.beam_section_list[sec_num[i]] for i in range(self.n_column,self.nm)],dtype=float)
		section_dummy = np.array([self.column_section_list[sec_num[i]] for i in range(self.n_column)]+[self.beam_section_list[sec_num[i]] for i in range(self.n_column,self.nm) for j in self.true_to_dummy_edge[i]],dtype=float)

		disp_dummy = [np.zeros((self.nk_dummy,3),dtype=float) for i in range(3)]
		stress_dummy = [np.zeros((self.nm_dummy,3),dtype=float) for i in range(3)]

		# update load
		load_dummy = []
		load_dummy.append(compute_load(self.NX,self.NY,self.span,self.height,section_dummy,self.length_dummy))
		load_dummy.append(np.copy(load_dummy[0]))
		load_dummy[1][:,0] *= -1
		load_dummy.append(np.copy(load_dummy[0]))
		load_dummy[2][:,0] = 0

		for loadcase in range(3):
			disp_dummy[loadcase], force_dummy = OpenSees.LinearAnalysis(self.NX,self.NY,self.node_dummy,self.connectivity_dummy,A=section_dummy[:,0],I=section_dummy[:,2],load=load_dummy[loadcase])
			stress_dummy[loadcase][:,0] = force_dummy[:,0]/section_dummy[:,0]
			stress_dummy[loadcase][:,1] = force_dummy[:,1]/section_dummy[:,4]
			stress_dummy[loadcase][:,2] = force_dummy[:,2]/section_dummy[:,4]
		
		# interlayer deformation angle
		deforms_r_c = np.zeros((self.n_column,3),dtype=float)
		deforms_r_b = np.zeros((self.nm-self.n_column,3),dtype=float)
		for i in range(3):
			deforms_r_c[:,i] = self.compute_deformation_ratio_column(disp_dummy[i])
			deforms_r_b[:,i] = self.compute_deformation_ratio_beam(disp_dummy[i])
		
		max_deform_r_c = np.max(deforms_r_c,axis=1)
		max_deform_r_b = np.max(deforms_r_b,axis=1)

		# maximum stress ratio
		alst_com_dummy, alst_ten_dummy, alst_ben_dummy = allowable_stress(self.nm,self.nm_dummy,self.n_column,section,self.length,self.E,self.DESIGN_STRENGTH_COLUMN,self.DESIGN_STRENGTH_BEAM,term='short')

		stress_r_dummy = np.zeros((self.nm_dummy,3),dtype=float)

		# short-term
		for i in range(2):
			stress_r_dummy[:,i] = np.max(np.abs(stress_dummy[i][:,1:]),axis=1)/alst_ben_dummy
			stress_r_dummy[stress_dummy[i][:,0]<0,i] += np.abs(stress_dummy[i][:,0]/alst_com_dummy)[stress_dummy[i][:,0]<0]
			stress_r_dummy[stress_dummy[i][:,0]>=0,i] += np.abs(stress_dummy[i][:,0]/alst_ten_dummy)[stress_dummy[i][:,0]>=0]

		# long-term
		stress_r_dummy[:,2] = np.max(np.abs(stress_dummy[2][:,1:]),axis=1)/(alst_ben_dummy/1.5)
		stress_r_dummy[stress_dummy[2][:,0]<0,2] += np.abs(stress_dummy[2][:,0]/(alst_com_dummy/1.5))[stress_dummy[2][:,0]<0]
		stress_r_dummy[stress_dummy[2][:,0]>=0,2] += np.abs(stress_dummy[2][:,0]/(alst_ten_dummy/1.5))[stress_dummy[2][:,0]>=0]

		max_stress_r_dummy = np.max(stress_r_dummy,axis=1)
		max_stress_r = np.zeros(self.nm,dtype=float)
		for i in range(self.nm):
			max_stress_r[i] = np.max(max_stress_r_dummy[self.true_to_dummy_edge[i]])

		# update cof
		Zp = np.array([self.column_plastic_section_modulus[sec_num[i]] for i in range(self.n_column)]+[self.beam_plastic_section_modulus[sec_num[i]] for i in range(self.n_column,self.nm)],dtype=float)
		cofs = np.zeros((self.nk,3))
		for i in range(3):
			cofs[:,i] = compute_cof(self.NX,self.NY,self.nk,self.n_column,self.DESIGN_STRENGTH_COLUMN,self.DESIGN_STRENGTH_BEAM,Zp,stress_dummy[i]) 
		min_cof = np.min(cofs,axis=1)

		# non-linear analysis
		if np.max(max_stress_r) > 1.0:
			hinge = None
			collapse = True
			max_deform_r_c_ultimate = None
		else:
			Zp_dummy = np.array([self.column_plastic_section_modulus[sec_num[i]] for i in range(self.n_column)]+[self.beam_plastic_section_modulus[sec_num[i]] for i in range(self.n_column,self.nm) for j in self.true_to_dummy_edge[i]],dtype=float)
			Sy_dummy = np.array([self.DESIGN_STRENGTH_COLUMN for i in range(self.n_column)]+[self.DESIGN_STRENGTH_BEAM for i in range(self.n_column,self.nm_dummy)],dtype=float)
			H_dummy = np.array([self.sec_num[i]*0.5/1000 for i in range(self.n_column)]+[self.sec_num[i]*0.5/1000 for i in range(self.n_column,self.nm) for j in self.true_to_dummy_edge[i]],dtype=float)

			ultimate_load = []
			collapse = False
			deforms_r_c_ultimate = np.zeros((self.n_column,2),dtype=float)
			for i in range(2):
				ultimate_load.append(np.copy(load_dummy[i]))
				ultimate_load[i][:,0] *= 1.5 # 1st design: C0 = 0.2, 2nd design: C0=1.0 and Ds = 0.3
				d,f,hinge = OpenSees.NonlinearAnalysis(self.NX,self.NY,self.node_dummy,self.connectivity_dummy,A=section_dummy[:,0],I=section_dummy[:,2],Zp=Zp_dummy,Sy=Sy_dummy,H=H_dummy,load=ultimate_load[i])
				deforms_r_c_ultimate[:,i] = self.compute_deformation_ratio_column(d)

			max_deform_r_c_ultimate = np.max(deforms_r_c_ultimate,axis=1)
			if np.max(max_deform_r_c_ultimate) > self.allowable_column_ultimate_disp:
				collapse = True

		# initialize node and edge inputs
		v = self.update_v(min_cof,v=v)
		w = self.update_w(sec_num,max_stress_r,max_deform_r_c,max_deform_r_b,w=w)

		volume = np.sum(self.length*np.array([self.column_section_list[self.sec_num[i]] for i in range(self.n_column)]+[self.beam_section_list[self.sec_num[i]] for i in range(self.n_column,self.nm)],dtype=float)[:,0])

		return v,w,disp_dummy[0],min_cof,max_deform_r_c,max_deform_r_b,max_stress_r,load_dummy[0],collapse,hinge,max_deform_r_c_ultimate,volume

	def compute_deformation_ratio_column(self,disp_dummy):
		'''
		disp_dummy[nk,3]: 0:x, 1:y, 2:Θ
		'''
		deformation = disp_dummy[self.connectivity_dummy[:self.n_column,1],0] - disp_dummy[self.connectivity_dummy[:self.n_column,0],0]
		deformation_ratio = np.abs(deformation/self.length[:self.n_column]) # (deformation)/(column length)
		return deformation_ratio

	def compute_deformation_ratio_beam(self,disp_dummy):
		'''
		disp_dummy[nk,3]: 0:x, 1:y, 2:Θ
		'''
		deformation = np.zeros((self.nm-self.n_column,2),dtype=float)
		for i in range(self.n_column,self.nm): # vertical deformation at the center from both tips
			deformation[i-self.n_column] = disp_dummy[self.connectivity_dummy[self.true_to_dummy_edge[i],1],1] - disp_dummy[self.connectivity_dummy[self.true_to_dummy_edge[i],0],1]
		deformation_ratio = np.max(np.abs(deformation),axis=1)/self.length[self.n_column:] # (deformation)/(beam length)
		return deformation_ratio

	def update_v(self,min_cof,v=None):
		'''
		0: 1 if the node is supported, else 0
		1: 1 if top nodes, else 0
		2: 1 if lateral boundary nodes, else 0
		3: 1.0 / column-to-beam strength ratio (among all load cases, maximum value is 2.0)
		'''
		if v is None:
			v=np.zeros((self.nk,4),dtype=np.float32)
			v[:self.NX+1,0] = 1.0
			v[self.nk-self.NX-1:,1] = 1.0
			v[np.logical_or(np.arange(self.nk)%(self.NX+1)==0,np.arange(self.nk)%(self.NX+1)==self.NX),2] = 1.0

		v[:,3] = np.tanh(1.0/min_cof)

		return v

	def update_w(self,sec_num,max_stress_ratio,max_deformation_ratio_column,max_deformation_ratio_beam,w=None):
		'''
		0        : 1 if column, else 0
		1        : 1 if beam, else 0
		2        : member length[m] / 15[m]
		3        : member A / A_max
		4        : member Iz / Iz_max
		5        : member Iy / Iy_max
		6        : member Zz / Zz_max
		7        : member A_next / A_max
		8        : member Iz_next / Iz_max
		9        : member Iy_next / Iy_max
		10        : member Zz_next / Zz_max
		11        : stress safety ratio (all the loads, maximum value is 2.0)
		12        : displacement safety ratio (all the loads, maximum value is 2.0)
		'''
		if w is None:
			w=np.zeros((self.nm,13),dtype=np.float32)
			## 0: column
			w[:,0] = 1.0-self.member_type
			## 1: beam
			w[:,1] = self.member_type
			## 2: length
			w[:,2] = self.length/15.0
		## 3-8: section
		sec = np.array([self.column_section_list[sec_num[i]] for i in range(self.n_column)]+[self.beam_section_list[sec_num[i]] for i in range(self.n_column,self.nm)],dtype=np.float32)
		if self.mode == 'dec':
			sec_num_next = np.clip(sec_num - 50,200,1000) # !!! reducing size: sec_num-50
		elif self.mode == 'inc':
			sec_num_next = np.clip(sec_num + 50,200,1000) # !!! increasing size: sec_num+50
		sec_next = np.array([self.column_section_list[sec_num_next[i]] for i in range(self.n_column)]+[self.beam_section_list[sec_num_next[i]] for i in range(self.n_column,self.nm)],dtype=np.float32)

		w[:,3] = sec[:,0] # A
		w[:,7] = sec_next[:,0] # A_next
		w[:self.n_column,[3,7]] /= self.column_section_list[1000][0]
		w[self.n_column:,[3,7]] /= self.beam_section_list[1000][0]
		w[:,4] = sec[:,2] # Iz
		w[:,8] = sec_next[:,2] # Iz_next
		w[:self.n_column,[4,8]] /= self.column_section_list[1000][2]
		w[self.n_column:,[4,8]] /= self.beam_section_list[1000][2]
		w[:,5] = sec[:,3] # Iy
		w[:,9] = sec_next[:,3] # Iy_next
		w[:self.n_column,[5,9]] /= self.column_section_list[1000][3]
		w[self.n_column:,[5,9]] /= self.beam_section_list[1000][3]
		w[:,6] = sec[:,4] # Zz
		w[:,10] = sec_next[:,4] # Zz_next
		w[:self.n_column,[6,10]] /= self.column_section_list[1000][4]
		w[self.n_column:,[6,10]] /= self.beam_section_list[1000][4]

		## 11: stress safety ratio
		w[:,11] = np.tanh(max_stress_ratio)
		## 12: displacement safety ratio
		w[:self.n_column,12] =  np.tanh(np.abs(max_deformation_ratio_column/self.allowable_column_disp))
		w[self.n_column:,12] =  np.tanh(np.abs(max_deformation_ratio_beam/self.allowable_beam_disp))

		return w

	def check_feasibility(self,stress_ratio,cofs,deform_r_c,deform_r_b,collapse):
		if not collapse and np.all(stress_ratio<1.0) and np.all(cofs[:-(self.NX+1)]>1.0) and np.all(deform_r_c<self.allowable_column_disp) and np.all(deform_r_b<self.allowable_beam_disp):
			feasible = True
		else:
			feasible = False
		return feasible

	def render(self,mode='section',show=False,title=None):
		'''
		mode = 'shape'
		  No annotation is provided. Just nodes and lines are illustrated. 
		mode = 'section'
		  In addition to 'shape' illustration, section numbers are provided.
		mode = 'ratio'
		  In addition to 'shape' illustration, safety ratio of members and column-to-beam strength ratio of nodes are provided.
		mode = 'disp'
		  In addition to 'shape' illustration, displacements ratio of columns and beams are provided.
		'''
		# state update to retrieve safety ratio of members
		self.v,self.w,disp_dummy,min_cof,max_deform_r_c,max_deform_r_b,max_ratio,load_dummy,_,hinge,max_deform_r_c_ultimate,_ = self.update_state(self.sec_num)
		
		# member info
		l_width, l_color, l_text = [],[],[]

		if mode == 'shape':
			node2d = self.node_dummy[:,0:2] + disp_dummy[:,0:2] * 100
			for i in range(self.nm):
				for j in self.true_to_dummy_edge[i]:
					l_width.append(self.sec_num[i])

		elif mode == 'section':
			node2d = np.copy(self.node[:,0:2])
			for i in range(self.nm):
				l_color.append("black")# if i != self.selected_action else "red")
				l_text.append('{0}'.format(self.sec_num[i]))
				l_width.append(self.sec_num[i])
		
		elif mode == 'disp':
			node2d = np.copy(self.node[:,0:2])
			for i in range(self.nm):
				l_width.append(self.sec_num[i])
				if i < self.n_column: 
					if i % (self.NX+1) == 0: # leftmost column
						l_color.append("orange" if max_deform_r_c[i]>(self.allowable_column_disp) else "black")
						l_text.append('{:.2f}'.format(max_deform_r_c[i]/self.allowable_column_disp))
					else: # other column
						l_color.append("black")
						l_text.append('')
				else: # beam
					l_color.append("orange" if max_deform_r_b[i-self.n_column]>(self.allowable_beam_disp) else "black")
					l_text.append('{:.2f}'.format(max_deform_r_b[i-self.n_column]/self.allowable_beam_disp))

		elif mode == 'stress':
			node2d = np.copy(self.node[:,0:2])
			for i in range(self.nm):
				l_width.append(self.sec_num[i])
				l_color.append("red" if max_ratio[i]>1.0 else "black")
				l_text.append('{:.2f}'.format(max_ratio[i]))

		elif mode == 'COF':
			node2d = np.copy(self.node[:,0:2])
			n_color, n_text = ["gray" for i in range(self.nk)], ["" for i in range(self.nk)]
			for i in range(self.nk):
				if i < self.NX+1: # bottom nodes
					n_color[i] = "black"
					n_text[i] = ''
				elif i > self.nk-self.NX-2: # upper nodes
					n_color[i] = "black"
					n_text[i] = ''
				else: # middle nodes
					n_color[i] = "red" if min_cof[i]<1.0 else "gray"
					n_text[i] = '{:.2f}'.format(min_cof[i])
			for i in range(self.nm):
				l_width.append(self.sec_num[i])
			
		elif mode == 'load':
			node2d = self.node_dummy[:,0:2]# + disp_dummy[:,0:2] * 50
			n_color, n_text = ["gray" for i in range(self.nk_dummy)], ["" for i in range(self.nk_dummy)]
			for i in range(self.nm):
				for j in self.true_to_dummy_edge[i]:
					l_width.append(self.sec_num[i])
			vector2d = load_dummy[:,0:2]
			for i in range(self.nk_dummy):
				n_text.append('{:.1f}'.format(load_dummy[i,0]))
		
		elif mode == 'hinge':
			node2d = self.node_dummy[:,0:2]
			for i in range(self.nm):
				for j in self.true_to_dummy_edge[i]:
					l_width.append(self.sec_num[i])

		elif mode == 'ultimate':
			node2d = np.copy(self.node[:,0:2])
			for i in range(self.nm):
				l_width.append(self.sec_num[i]) # leftmost column
				if i < self.n_column and i % (self.NX+1) == 0:
					l_color.append("red" if np.all(max_deform_r_c_ultimate == None) or max_deform_r_c_ultimate[i]>(self.allowable_column_ultimate_disp) else "black")
					l_text.append('{:}'.format(np.inf) if np.all(max_deform_r_c_ultimate == None) else '{:.2f}'.format(max_deform_r_c_ultimate[i]/self.allowable_column_ultimate_disp))
				else: # other member
					l_color.append("black")
					l_text.append('')			
		
		if mode == 'shape':
			outfile = Plotter.Draw(node2d,self.connectivity_dummy,line_width=l_width,node_color=None,line_color=None,node_text=None,line_text=None,name=self.steps,show=show,title=title)
		elif mode == 'section':
			outfile = Plotter.Draw(node2d,self.connectivity,line_width=l_width,node_color=None,line_color=l_color,node_text=None,line_text=l_text,name=self.steps,show=show,title=title)
		elif mode == 'stress':
			outfile = Plotter.Draw(node2d,self.connectivity,line_width=l_width,node_color=None,line_color=l_color,node_text=None,line_text=l_text,name=self.steps,show=show,title=title)
		elif mode == 'COF':
			outfile = Plotter.Draw(node2d,self.connectivity,line_width=l_width,node_color=n_color,line_color=None,node_text=n_text,line_text=None,name=self.steps,show=show,title=title)
		elif mode == 'disp' or mode == 'ultimate':
			outfile = Plotter.Draw(node2d,self.connectivity,line_width=l_width,node_color=None,line_color=l_color,node_text=None,line_text=l_text,name=self.steps,show=show,title=title)
		elif mode == 'load':
			outfile = Plotter.Draw(node2d,self.connectivity_dummy,line_width=l_width,node_color=None,line_color=None,node_text=n_text,line_text=None,vector=vector2d,name=self.steps,show=show,title=title)
		elif mode == 'hinge':
			outfile = Plotter.Draw(node2d,self.connectivity_dummy,line_width=l_width,node_color=None,line_color=None,node_text=None,line_text=None,vector=None,hinge=hinge,name=self.steps,show=show,title=title)
		else:
			raise Exception("Unexpected mode selection.")

		return outfile

	def func(self,x):
		### if x = 1: sec_num = 1000
		### if x = 0: sec_num = 200

		self.sec_num = 200+50*np.array(x*17-1.0e-10,dtype=int) # 17 = number of member types (200,250,...,1000)

		for i in range(self.n_column-(self.NX+1)):
			max_sec_above = np.max(self.sec_num[self.dependency[0][i]])
			if self.sec_num[i] < max_sec_above:
				self.sec_num[i] = max_sec_above

		self.v,self.w,_,cofs,deform_r_c,deform_r_b,stress_ratio,_,collapse,_,_,volume = self.update_state(self.sec_num)
		self.feasible = self.check_feasibility(stress_ratio,cofs,deform_r_c,deform_r_b,collapse)

		if self.feasible:
			f = volume
		else:
			Max_stress_ratio = np.max(stress_ratio)
			Max_disp_ratio = np.max(np.concatenate([deform_r_c/(1/200),deform_r_b/(1/300)]))
			Min_cof = np.min(cofs)

			p1 = np.clip(Max_stress_ratio-1.0,0.0,None)
			p2 = np.clip(Max_disp_ratio-1.0,0.0,None)
			p3 = np.clip(1.0-Min_cof,0.0,None)
			p4 = float(collapse)

			f = volume + p1*1.0e3 + p2*1.0e3 + p3*1.0e3 + p4*1.0e3

		return f, self.feasible

# class FrameSection_Increment(FrameSection):

# 	action_type = "^"

# 	def reset(self,test=False):
# 		return self.reset_original(200,test)

# 	def step(self, action):
# 		return self.step_original(action,"^")

# 	def _is_done(self,feasible):
# 		if self.steps > self.MAX_STEPS or feasible:
# 			return True
# 		else:
# 			return False
	
# class FrameSection_Decrement(FrameSection):

# 	action_type = "v"

# 	def reset(self,test=False):
# 		return self.reset_original(1000,test)

# 	def step(self, action):
# 		return self.step_original(action,"v")

# 	def _is_done(self,feasible):
# 		if self.steps > self.MAX_STEPS or not feasible:
# 			return True
# 		else:
# 			return False

# class FrameSection_Optimization(FrameSection):

# 	def reset(self,test=False):
# 		return self.reset_original(500,test)

# 	def step(self,action,action_type):
# 		return self.step_original(action, action_type)

# 	def _is_done(self,feasible):
# 		if self.steps > 200:
# 			return True
# 		else:
# 			return False

