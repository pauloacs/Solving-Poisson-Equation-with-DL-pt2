from ctypes import py_object
import matplotlib
from numpy.core.defchararray import array
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from numba import njit
import tensorflow as tf
import os
import shutil
import time
import h5py
import numpy as np
import scipy.spatial.qhull as qhull

import pickle as pk
import itertools
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath

from shapely.geometry import MultiPoint
from scipy.spatial import distance
import scipy
import scipy.ndimage as ndimage
from scipy import interpolate

class Evaluation():
	def __init__(self, delta, shape, overlap, var_p, var_in, dataset_path, model_path, max_num_PC, standardization_method):
		"""
		Initialize Evaluation class. 

        Args:
            delta (float): 
            shape (int): 
            overlap (float): 
			var_p (float): 
			var_in (float): 
			dataset_path (str):
			model_path (str):
        """
		self.delta = delta
		self.shape = shape
		self.overlap = overlap
		self.var_in = var_in
		self.var_p = var_p
		self.dataset_path = dataset_path
		self.standardization_method = standardization_method

		maxs = np.loadtxt('maxs')
		#maxs_PCA = np.loadtxt('maxs_PCA')

		self.max_abs_Ux, self.max_abs_Uy, self.max_abs_dist, self.max_abs_p = maxs[0], maxs[1], maxs[2], maxs[3]
		#self.max_abs_input_PCA, self.max_abs_output_PCA = maxs_PCA[0], maxs_PCA[1]

		#### loading the model #######
		self.model = tf.keras.models.load_model(model_path)
		print(self.model.summary())
		
		### loading the pca matrices for transformations ###
		self.pcainput = pk.load(open("ipca_input.pkl",'rb'))
		self.pcap = pk.load(open("ipca_p.pkl",'rb'))

		self.pc_p = np.argmax(self.pcap.explained_variance_ratio_.cumsum() > self.var_p) if np.argmax(self.pcap.explained_variance_ratio_.cumsum() > self.var_p) > 1 and np.argmax(self.pcap.explained_variance_ratio_.cumsum() > self.var_p) <= max_num_PC else max_num_PC
		self.pc_in = np.argmax(self.pcainput.explained_variance_ratio_.cumsum() > self.var_in) if np.argmax(self.pcainput.explained_variance_ratio_.cumsum() > self.var_in) > 1 and np.argmax(self.pcainput.explained_variance_ratio_.cumsum() > self.var_in) <= max_num_PC else max_num_PC


	def interp_weights(self, xyz, uvw):
		"""
		Gets the interpolation's verticies and weights from xyz to uvw.

		Args:
			xyz (NDArray): Original array of coordinates.
			uvw (NDArray): Target array of coordinates
		"""
		d = 2 #2d interpolation
		tri = qhull.Delaunay(xyz)
		simplex = tri.find_simplex(uvw)
		vertices = np.take(tri.simplices, simplex, axis=0)
		temp = np.take(tri.transform, simplex, axis=0)
		delta = uvw - temp[:, d]
		bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
		return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

	def interpolate(self, values, vtx, wts):
		"""
		Interpolate based on previously computed vertices (vtx) and weights (wts).

		Args:
			values (NDArray): Array of values to interpolate.
			vtx (NDArray): Array of interpolation vertices.
			wts (NDArray): Array of interpolation weights.
		"""
		return np.einsum('nj,nj->n', np.take(values, vtx), wts)

	def interpolate_fill(self, values, vtx, wts, fill_value=np.nan):
		"""
		Interpolate based on previously computed vertices (vtx) and weights (wts) and fill.

		Args:
			values (NDArray): Array of values to interpolate.
			vtx (NDArray): Array of interpolation vertices.
			wts (NDArray): Array of interpolation weights.
			fill_value (float): Value used to fill.
		"""
		ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
		ret[np.any(wts < 0, axis=1)] = fill_value
		return ret

	
	def create_uniform_grid(self, x_min, x_max, y_min, y_max):
		"""
		Creates an uniform 2D grid (should envolve every cell of the mesh).

        Args:
            x_min (float): The variable name is self-explanatory.
            x_max (float): The variable name is self-explanatory.
            y_min (float): The variable name is self-explanatory.
			y_max (float): The variable name is self-explanatory.
		"""
		X0 = np.linspace(x_min + self.delta/2 , x_max - self.delta/2 , num = int(round( (x_max - x_min)/self.delta )) )
		Y0 = np.linspace(y_min + self.delta/2 , y_max - self.delta/2 , num = int(round( (y_max - y_min)/self.delta )) )

		XX0, YY0 = np.meshgrid(X0,Y0)
		return XX0.flatten(), YY0.flatten()

	#@njit(nopython = True)  #much faster using numba.njit but is giving an error
	def index(self, array, item):
		"""
		Finds the index of the first element equal to item.

		Args:
			array (NDArray):
			item (float):
		"""
		for idx, val in np.ndenumerate(array):
			if val == item:
				return idx
			# else:
			# 	return None
		# If no item was found return None, other return types might be a problem due to
		# numbas type inference.

	def read_dataset(self, path, sim, time):
		"""
		Reads dataset and splits it into the internal flow data (data) and boundary data.

		Args:
			path (str): Path to hdf5 dataset
			sim (int): Simulation number.
			time (int): Time frame.
		"""
		with h5py.File(path, "r") as f:
			data = f["sim_data"][sim:sim+1,time:time+1, ...]
			top_boundary = f["top_bound"][sim:sim+1, time:time+1 , ...]
			obst_boundary = f["obst_bound"][sim:sim+1, time:time+1 , ...]
		return data, top_boundary, obst_boundary

	def computeOnlyOnce(self, sim):
		"""
		Performs interpolation from the OF grid (corresponding to the mesh cell centers),
		saves the intepolation vertices and weights and computes the signed distance function (sdf).

		Args:
			sim (int): Simulation number.
		"""
		time = 0
		data, top_boundary, obst_boundary = self.read_dataset(self.dataset_path, sim , time)

		self.indice = self.index(data[0,0,:,0] , -100.0 )[0]


		x_min = round(np.min(data[0,0,...,:self.indice,3]),3) 
		x_max = round(np.max(data[0,0,...,:self.indice,3]),3) 

		y_min = round(np.min(data[0,0,...,:self.indice,4]),3)  #- 0.3
		y_max = round(np.max(data[0,0,...,:self.indice,4]),3)  #+ 0.3

		######### -------------------- Assuming constant mesh, the following can be done out of the for cycle ------------------------------- ##########

		X0, Y0 = self.create_uniform_grid(x_min, x_max, y_min, y_max)
		self.X0 = X0
		self.Y0 = Y0
		xy0 = np.concatenate((np.expand_dims(X0, axis=1),np.expand_dims(Y0, axis=1)), axis=-1)
		points = data[0,0,:self.indice,3:5] #coordinates
		self.vert, self.weights = self.interp_weights(points, xy0) #takes ~100% of the interpolation cost and it's only done once for each different mesh/simulation case

		# boundaries indice
		indice_top = self.index(top_boundary[0,0,:,0] , -100.0 )[0]
		top = top_boundary[0,0,:indice_top,:]
		self.max_x, self.max_y = np.max([(top[:,0]).max(), x_max]), np.min([(top[:,1]).max(), x_max])
		self.min_x, self.min_y = np.max([(top[:,0]).min(), x_min]), np.min([(top[:,1]).min(), y_min])

		is_inside_domain = ( xy0[:,0] <= self.max_x)  * ( xy0[:,0] >= self.min_x ) * ( xy0[:,1] <= self.max_y ) * ( xy0[:,1] >= self.min_y ) #rhis is just for simplification

		indice_obst = self.index(obst_boundary[0,0,:,0] , -100.0 )[0]
		obst = obst_boundary[0,0,:indice_obst,:]

		obst_points = MultiPoint(obst)

		# This only works for convex geometries
		hull = obst_points.convex_hull  
		hull_pts = hull.exterior.coords.xy  
		hull_pts = np.c_[hull_pts[0], hull_pts[1]]

		path = mpltPath.Path(hull_pts)
		is_inside_obst = path.contains_points(xy0)

		domain_bool = is_inside_domain * ~is_inside_obst

		top = top[0:top.shape[0]:2,:]   #if this has too many values, using cdist can crash the memmory since it needs to evaluate the distance between ~1M points with thousands of points of top
		obst = obst[0:obst.shape[0]:2,:]

		sdf = np.minimum(distance.cdist(xy0,obst).min(axis=1), distance.cdist(xy0,top).min(axis=1) ) * domain_bool

		#div defines the sliding window vertical and horizontal displacements
		div = 1 

		self.grid_shape_y = int(round((y_max-y_min)/self.delta)) #+1
		self.grid_shape_x = int(round((x_max-x_min)/self.delta)) #+1

		i = 0
		j = 0

		#arrange data in array: #this can be put outside the j loop if the mesh is constant 

		x0 = np.min(X0)
		y0 = np.min(Y0)
		dx = self.delta
		dy = self.delta

		indices= np.zeros((X0.shape[0],2))
		obst_bool = np.zeros((self.grid_shape_y,self.grid_shape_x,1))
		self.sdfunct = np.zeros((self.grid_shape_y,self.grid_shape_x,1))

		p = data[i,j,:self.indice,2:3] #values
		p_interp = self.interpolate_fill(p, self.vert, self.weights) 

		for (step, x_y) in enumerate(xy0):  
			if domain_bool[step] * (~np.isnan(p_interp[step])) :
				jj = int(round((x_y[...,0] - x0) / dx))
				ii = int(round((x_y[...,1] - y0) / dy))

				indices[step,0] = ii
				indices[step,1] = jj
				self.sdfunct[ii,jj,:] = sdf[step]
				obst_bool[ii,jj,:]  = int(1)

		self.indices = indices.astype(int)

		self.pred_minus_true = []
		self.pred_minus_true_squared = []

		self.pred_minus_true_p = []
		self.pred_minus_true_squared_p = []

		return 0

	def assemble_prediction(self, array, indices_list, n_x, n_y, apply_filter, shape_x, shape_y):
		"""
		Reconstructs the flow domain based on squared blocks.
		In the first row the correction is based on the outlet fixed value BC.
		
		In the following rows the correction is based on the overlap region at the top of each new block.
		This correction from the top ensures better agreement between different rows, leading to overall better results.

		Args:
			array
			indices_list
			n_x
			n_y
			apply_filter
			shape_x
			shape_y 
		"""
		overlap = self.overlap
		shape = self.shape
		Ref_BC = self.Ref_BC

		result_array = np.empty(shape=(shape_y, shape_x))

		## Array to store the average pressure in the overlap region with the next down block
		BC_ups = np.zeros(n_x+1)

		# i index where the lower blocks are located
		p_i = shape_y - ( (shape-overlap) * n_y + shape )
		
		# j index where the left-most blocks are located
		p_j = shape_x - ( (shape - overlap) * n_x + shape )

		result = result_array

		## Loop over all the blocks and apply corrections to ensure consistency between overlapping blocks
		for i in range(self.x_array.shape[0]):

			idx_i, idx_j = indices_list[i]
			flow_bool = self.x_array[i,:,:,2]
			pred_field = array[i,...]

			## FIRST row
			if idx_i == 0:

				## Calculating correction to be applied
				if i == 0: 
 					## First correction - based on the outlet fixed pressure boundary condition
					BC_coor = np.mean(pred_field[:,-1][flow_bool[:,-1]!=0]) - Ref_BC  # i = 0 sits outside the inclusion zone
				else:
					BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
					BC_coor = np.mean(pred_field[:,-overlap:][flow_bool[:,-overlap:]!=0]) - BC_ant_0
				if idx_j == 0:
					intersect_zone_limit = overlap - p_j
					BC_ant_0 = np.mean(old_pred_field[:, :intersect_zone_limit][flow_bool[:, :intersect_zone_limit] !=0]) 
					BC_coor = np.mean(pred_field[:, -intersect_zone_limit:][flow_bool[:, -intersect_zone_limit:]!=0]) - BC_ant_0

				## Applying correction
				pred_field -= BC_coor

				## Storing upward average pressure
				BC_ups[idx_j] = np.mean(pred_field[-overlap:,:][flow_bool[-overlap:,:] !=0])

			## MIDDLE rows
			elif idx_i != n_y + 1:

				## Calculating correction to be applied
				if np.isnan(BC_ups[idx_j]): #### THIS PART IS NOT WORKING WELL ... CORRECT THIS!!
					if idx_j == 0:
						intersect_zone_limit = overlap - p_j
						BC_ant_0 = np.mean(old_pred_field[:, :intersect_zone_limit][flow_bool[:, :intersect_zone_limit] !=0]) 
						BC_coor = np.mean(pred_field[:, -intersect_zone_limit:][flow_bool[:, -intersect_zone_limit:]!=0]) - BC_ant_0
					elif idx_j == n_x:
						# Here it always needs to be corrected from above to keep consistency
						BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
						BC_coor = np.mean(pred_field[:overlap,:][flow_bool[:overlap,:]!=0]) - BC_ups[idx_j]
					else:
						BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
						BC_coor = np.mean(pred_field[:,-overlap:][flow_bool[:,-overlap:]!=0]) - BC_ant_0											
				else:
					BC_coor = np.mean(pred_field[:overlap,:][flow_bool[:overlap,:]!=0]) - BC_ups[idx_j]
				
				## Applying correction
				pred_field -= BC_coor

				## Storing upward average pressure
				BC_ups[idx_j] = np.mean(pred_field[-overlap:,:][flow_bool[-overlap:,:] !=0])
				
				## Value stored to be used in the last row depends on p_i
				if idx_i == n_y:
					BC_ups[idx_j] = np.mean(pred_field[-(shape-p_i):,:][flow_bool[-(shape-p_i):,:] !=0])
			
			## LAST row
			else:

				## Calculating correction to be applied

				## In the last column the correction needs to be from above to keep consistency (with BC_ups)
				if idx_j == n_x:
					BC_coor = np.mean(pred_field[-p_i-overlap:-p_i,:][flow_bool[-p_i-overlap:-p_i,:]!=0]) - BC_ups[idx_j]
				else:
									
					#up
					y_0 = -p_i - overlap
					y_f = -p_i
					n_up_non_nans = (flow_bool[y_0:y_f,:]!=0).sum()
					# right side
					x_0 = shape_x -shape - (n_x-1)*(shape-overlap)
					n_right_non_nans = (flow_bool[:, x_0:]!=0).sum()

					# Give preference to "up" or "right" correction???
					## Giving it to "up" because it is being done everywhere else
					## only switch method if in the overlap region more than 90% of the values are NANs
					
					if (n_up_non_nans)/128**2 > 0.9:
						if idx_j == 0:
							intersect_zone_limit = overlap - p_j
							BC_ant_0 = np.mean(old_pred_field[:, :intersect_zone_limit][flow_bool[:, :intersect_zone_limit] !=0]) 
							BC_coor = np.mean(pred_field[:, -intersect_zone_limit:][flow_bool[:, -intersect_zone_limit:]!=0]) - BC_ant_0
						else:
							BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
							BC_coor = np.mean(pred_field[:,-overlap:][flow_bool[:,-overlap:]!=0]) - BC_ant_0								
					else:
						BC_coor = np.mean(pred_field[:-p_i,:][flow_bool[:-p_i,:]!=0]) - BC_ups[idx_j]
				
				## Applying the correction
				pred_field -= BC_coor
				
			old_pred_field = pred_field
			
			## Last reassembly step:
			## Assigning the block to the right location in the flow domain
			if [idx_i, idx_j] == [n_y + 1, 0]:
				result[-p_i:shape_y , 0:shape] = pred_field[-p_i:]
			elif idx_j == 0:
				result[(shape-overlap) * idx_i:(shape-overlap) * idx_i + shape, 0:shape] = pred_field
			elif idx_i == (n_y + 1):
				idx_j = n_x - idx_j
				result[-p_i:, shape_x -shape - idx_j*(shape-overlap) :  shape_x- idx_j*(shape-overlap)] = pred_field[-p_i:]
			else:
				idx_j = n_x - idx_j
				result[(shape-overlap) * idx_i:(shape-overlap) * idx_i + shape, shape_x -shape - idx_j*(shape-overlap) : shape_x- idx_j*(shape-overlap)] = pred_field
				
		result -= np.mean( 3* result[:,-1] - result[:,-2] )/3


		################### this applies a gaussian filter to remove boundary artifacts #################

		if apply_filter:
			result = ndimage.gaussian_filter(result, sigma=(10, 10), order=0)

		return result

	def timeStep(self, sim, time, plot_intermediate_fields, save_plots, show_plots, apply_filter):
		"""

		Args:
			sim (int):
			time (int):
			save_plots (bool):
			show_plots (bool):
			apply_filter
		"""
	
		data, top_boundary, obst_boundary = self.read_dataset(self.dataset_path, sim , time)
		i = 0
		j = 0
		
		Ux =  data[i,j,:self.indice,0:1] #values
		Uy =  data[i,j,:self.indice,1:2] #values
		delta_p = data[i,j,:self.indice,7:8] #values
		delta_Ux = data[i,j,:self.indice,5:6] #values
		delta_Uy = data[i,j,:self.indice,6:7] #values
		p = data[i,j,:self.indice,2:3] #values

		U_max_norm = np.max(np.sqrt(np.square(Ux) + np.square(Uy)))

		delta_p_adim = delta_p / pow(U_max_norm,2.0) 
		delta_Ux_adim = delta_Ux/U_max_norm 
		delta_Uy_adim = delta_Uy/U_max_norm

		delta_p_interp = self.interpolate_fill(delta_p_adim, self.vert, self.weights)
		delta_Ux_interp = self.interpolate_fill(delta_Ux_adim, self.vert, self.weights)
		delta_Uy_interp = self.interpolate_fill(delta_Uy_adim, self.vert, self.weights)
		p_interp = self.interpolate_fill(p, self.vert, self.weights)

		grid = np.zeros(shape=(1, self.grid_shape_y, self.grid_shape_x, 5))

		grid[0,:,:,0:1][tuple(self.indices.T)] = delta_Ux_interp.reshape(delta_Ux_interp.shape[0],1)
		grid[0,:,:,1:2][tuple(self.indices.T)] = delta_Uy_interp.reshape(delta_Uy_interp.shape[0],1)
		grid[0,:,:,2:3] = self.sdfunct
		grid[0,:,:,3:4][tuple(self.indices.T)] = delta_p_interp.reshape(delta_p_interp.shape[0],1)
		grid[0,:,:,4:5][tuple(self.indices.T)] = p_interp.reshape(p_interp.shape[0],1)

		grid[np.isnan(grid)] = 0 #set any nan value to 0

		grid[0,:,:,0:1] = grid[0,:,:,0:1]/self.max_abs_Ux
		grid[0,:,:,1:2] = grid[0,:,:,1:2]/self.max_abs_Uy
		grid[0,:,:,2:3] = grid[0,:,:,2:3]/self.max_abs_dist
		grid[0,:,:,3:4] = grid[0,:,:,3:4]/self.max_abs_p

		## Block extraction
		x_list = []
		obst_list = []
		y_list = []
		indices_list = []

		overlap = self.overlap
		shape = self.shape

		n_x = int(np.ceil((grid.shape[2]-shape)/(shape - overlap )) )
		n_y = int((grid.shape[1]-shape)/(shape - overlap ))

		for i in range ( n_y + 2 ): #+1 b
			for j in range ( n_x +1 ):
				
				# going right to left
				x_0 = grid.shape[2] - j*shape + j*overlap - shape
				if j == n_x: x_0 = 0
				x_f = x_0 + shape

				y_0 = i*shape - i*overlap
				if i == n_y + 1: y_0 = grid.shape[1]-shape
				y_f = y_0 + shape

				x_list.append(grid[0:1, y_0:y_f, x_0:x_f, 0:3])
				y_list.append(grid[0:1, y_0:y_f, x_0:x_f, 3:4])

				indices_list.append([i, n_x - j])

		self.x_array = np.concatenate(x_list)
		self.y_array = np.concatenate(y_list)
		y_array = self.y_array
		N = self.x_array.shape[0]
		features = self.x_array.shape[3]
		
		for step in range(y_array.shape[0]):
			y_array[step,...,0][self.x_array[step,...,2] != 0] -= np.mean(y_array[step,...,0][self.x_array[step,...,2] != 0])

		x_array_flat = self.x_array.reshape((N, self.x_array.shape[1]*self.x_array.shape[2], features ))
		input_flat = x_array_flat.reshape((x_array_flat.shape[0],-1))

		y_array_flat = y_array.reshape((N, y_array.shape[1]*y_array.shape[2], 1))
		y_array_flat = y_array_flat.reshape((y_array_flat.shape[0],-1))

		input_transformed = self.pcainput.transform(input_flat)[:,:self.pc_in]
		print(' Total variance from input represented: ' + str(np.sum(self.pcainput.explained_variance_ratio_[:self.pc_in])))
		print(input_transformed.shape)

		y_transformed = self.pcap.transform(y_array_flat)[:,:self.pc_p]
		print(' Total variance from Obst_bool represented: ' + str(np.sum(self.pcap.explained_variance_ratio_[:self.pc_p])))

		if self.standardization_method == 'std':
			## Option 1: Standardization
			data = np.load('mean_std.npz')
			mean_in_loaded = data['mean_in']
			std_in_loaded = data['std_in']
			mean_out_loaded = data['mean_out']
			std_out_loaded = data['std_out']
			x_input = (input_transformed - mean_in_loaded) /std_in_loaded	
		elif self.standardization_method == 'min_max':
			## Option 2: Min-max scaling
			data = np.load('min_max_values.npz')
			min_in_loaded = data['min_in']
			max_in_loaded = data['max_in']
			min_out_loaded = data['min_out']
			max_out_loaded = data['max_out']
			x_input = (input_transformed - min_in_loaded) / (max_in_loaded - min_in_loaded)
		elif self.standardization_method == 'max_abs':
			## Option 3: Old method
			x_input = input_transformed/self.max_abs_input_PCA	
		else:
			raise("Standardization method not valid")

		comp = self.pcap.components_
		pca_mean = self.pcap.mean_

		res_concat = np.array(self.model(np.array(x_input)))

		if self.standardization_method == 'std':
			res_concat = (res_concat * std_out_loaded) + mean_out_loaded
		elif self.standardization_method == 'min_max':
			res_concat = res_concat * (max_out_loaded - min_out_loaded) + min_out_loaded
		elif self.standardization_method == 'max_abs':
			res_concat *= self.max_abs_output_PCA 
		else:
			raise("Standardization method not valid")

		res_flat_inv = np.dot(res_concat, comp[:self.pc_p, :]) + pca_mean	
		res_concat = res_flat_inv.reshape((res_concat.shape[0], shape, shape, 1)) 

		# Dimensionalize pressure field
		res_concat = res_concat#  * pow(delta_U_max_norm,2.0) * self.max_abs_p

		# the boundary condition is a fixed pressure of 0 at the output
		self.Ref_BC = 0 


		# performing the assembly process
		res = self.assemble_prediction(res_concat[...,0], indices_list, n_x, n_y, apply_filter, grid.shape[2], grid.shape[1])
		test_res = self.assemble_prediction(y_array[...,0], indices_list, n_x, n_y, apply_filter, grid.shape[2], grid.shape[1])
		
		################## ----------------//---------------####################################

		## use field=test_res to test the assembly algorith -> it should be almost perfect in that case

		field = res

		# Plotting the integrated pressure field
		fig, axs = plt.subplots(3,2, figsize=(65, 15))

		vmax = np.max(grid[0,:,:,3])
		vmin = np.min(grid[0,:,:,3])

		masked_arr = np.ma.array(field, mask=(grid[0,:,:,2] == 0))
		axs[0,0].set_title('delta_p predicted', fontsize = 15)
		cf = axs[0,0].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin )
		plt.colorbar(cf, ax=axs[0,0])

		masked_arr = np.ma.array(grid[0,:,:,3], mask=(grid[0,:,:,2] == 0))
		axs[1,0].set_title('CFD results', fontsize = 15)
		cf = axs[1,0].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin)
		plt.colorbar(cf, ax=axs[1,0])

		masked_arr = np.ma.array( np.abs(( grid[0,:,:,3] - field )/(np.max(grid[0,:,:,3]) -np.min(grid[0,:,:,3]))*100) , mask=(grid[0,:,:,2] == 0))
		axs[2,0].set_title('error in %', fontsize = 15)
		cf = axs[2,0].imshow(masked_arr, interpolation='nearest', cmap='jet', vmax = 10, vmin=0 )
		plt.colorbar(cf, ax=axs[2,0])
		
		# actual pressure fields

		# Infering p_t-1 from ref p and delta_p
		p_prev = grid[0,:,:,4] - grid[0,:,:,3]
		p_pred = p_prev + field

		masked_arr = np.ma.array(p_pred, mask=(grid[0,:,:,2] == 0))
		axs[0,1].set_title(r'Predicted pressure $p_{t-1} + delta_p$', fontsize = 15)
		cf = axs[0,1].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin )
		plt.colorbar(cf, ax=axs[0,1])

		masked_arr = np.ma.array(grid[0,:,:,4], mask=(grid[0,:,:,2] == 0))
		axs[1,1].set_title('Pressure (CFD)', fontsize = 15)
		cf = axs[1,1].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin)
		plt.colorbar(cf, ax=axs[1,1])

		masked_arr = np.ma.array( np.abs(( grid[0,:,:,4] - p_pred )/(np.max(grid[0,:,:,4]) -np.min(grid[0,:,:,4]))*100) , mask=(grid[0,:,:,2] == 0))

		axs[2,1].set_title('error in %', fontsize = 15)
		cf = axs[2,1].imshow(masked_arr, interpolation='nearest', cmap='jet', vmax = 2, vmin=0 )
		plt.colorbar(cf, ax=axs[2,1])


		if show_plots:
			plt.show()

		if save_plots:
			fig.savefig(f'plots/p_pred_sim{sim}t{time}.png')

		plt.close()

		############## ------------------//------------------##############################
		
		# This part is to output the error metrics

		true_mask = grid[0,:,:,3][grid[0,:,:,2] != 0]
		pred_mask = field[grid[0,:,:,2] != 0]
		norm = np.max(grid[0,:,:,3][grid[0,:,:,2] != 0]) - np.min(grid[0,:,:,3][grid[0,:,:,2] != 0])

		mask_nan = ~np.isnan( pred_mask  - true_mask )

		BIAS_norm = np.mean( (pred_mask  - true_mask )[mask_nan] )/norm * 100
		RMSE_norm = np.sqrt(np.mean( ( pred_mask  - true_mask )[mask_nan]**2 ))/norm * 100
		STDE_norm = np.sqrt( (RMSE_norm**2 - BIAS_norm**2) )
		

		print(f"""
		** Error in delta_p **

		normVal  = {norm} Pa
		biasNorm = {BIAS_norm:.2f}%
		stdeNorm = {STDE_norm:.2f}%
		rmseNorm = {RMSE_norm:.2f}%
		""")

		self.pred_minus_true.append( np.mean( (pred_mask  - true_mask )[mask_nan] )/norm )
		self.pred_minus_true_squared.append( np.mean( (pred_mask  - true_mask )[mask_nan]**2 )/norm**2 )
		
		## Error in p

		true_mask = grid[0,:,:,4][grid[0,:,:,2] != 0]
		pred_mask = p_pred[grid[0,:,:,2] != 0]
		norm = np.max(grid[0,:,:,4][grid[0,:,:,2] != 0]) - np.min(grid[0,:,:,4][grid[0,:,:,2] != 0])

		mask_nan = ~np.isnan( pred_mask  - true_mask )

		BIAS_norm = np.mean( (pred_mask  - true_mask )[mask_nan] )/norm * 100
		RMSE_norm = np.sqrt(np.mean( ( pred_mask  - true_mask )[mask_nan]**2 ))/norm * 100
		STDE_norm = np.sqrt( (RMSE_norm**2 - BIAS_norm**2) )


		print(f"""
		** Error in p **

		normVal  = {norm} Pa
		biasNorm = {BIAS_norm:.2f}%
		stdeNorm = {STDE_norm:.2f}%
		rmseNorm = {RMSE_norm:.2f}%
		""")

		self.pred_minus_true_p.append( np.mean( (pred_mask  - true_mask )[mask_nan] )/norm )
		self.pred_minus_true_squared_p.append( np.mean( (pred_mask  - true_mask )[mask_nan]**2 )/norm**2 )



		return 0

	def createGIF(self, n_sims, n_ts):
		
		####################### TO CREATE A GIF WITH ALL THE FRAMES ###############################
		filenamesp = []

		for sim in range(5):
			for time in range(5):
				filenamesp.append(f'plots/p_pred_sim{sim}t{time}.png') #hardcoded to get the frames in order

		import imageio

		with imageio.get_writer('plots/p_movie.gif', mode='I', duration =0.5) as writer:
			for filename in filenamesp:
				image = imageio.imread(filename)
				writer.append_data(image)
		######################## ---------------- //----------------- ###################



def call_SM_main(delta, model_name, shape, overlap_ratio, var_p, var_in, max_num_PC, dataset_path,	\
					plot_intermediate_fields, standardization_method, save_plots, show_plots, apply_filter, create_GIF, \
					n_sims, n_ts):

	### This creates a directory to save the plots
	path='plots/'

	try:
		shutil.rmtree(path)
	except OSError as e:
		print ("")

	os.makedirs(path)

	overlap = int(overlap_ratio*shape)
	
	Eval = Evaluation(delta, shape, overlap, var_p, var_in, dataset_path, model_name, max_num_PC, standardization_method)
	Eval.pred_minus_true = []
	Eval.pred_minus_true_squared = []

	# Simulations to use for evaluation
	# This points to the number of the simulation data in the dataset
	sims = list(range(n_sims))

	for sim in sims:
		Eval.computeOnlyOnce(sim)
		# Timesteps used for evaluation
		for time in range(n_ts):
			Eval.timeStep(sim, time, plot_intermediate_fields, save_plots, show_plots, apply_filter)

		BIAS_value = np.mean(Eval.pred_minus_true) * 100
		RMSE_value = np.sqrt(np.mean(Eval.pred_minus_true_squared)) * 100
		STDE_value = np.sqrt( RMSE_value**2 - BIAS_value**2 )

		BIAS_value_p = np.mean(Eval.pred_minus_true_p) * 100
		RMSE_value_p = np.sqrt(np.mean(Eval.pred_minus_true_squared_p)) * 100
		STDE_value_p = np.sqrt( RMSE_value_p**2 - BIAS_value_p**2 )

		print(f'''
		Average across the WHOLE simulation:

		** Error in delta_p **

		BIAS: {BIAS_value:.2f}%
		STDE: {STDE_value:.2f}%
		RMSE: {RMSE_value:.2f}%

		** Error in p **
		BIAS: {BIAS_value_p:.2f}%
		STDE: {STDE_value_p:.2f}%
		RMSE: {RMSE_value_p:.2f}%
		''')

	if create_GIF:
		self.createGIF(n_sims, n_ts)


if __name__ == '__main__':

	delta = 5e-3
	model_name = 'model_small-std-0.95.h5'
	shape = 128
	overlap_ratio = 0.25
	var_p = 0.95
	var_in = 0.95
	max_num_PC = 128
	dataset_path = '../dataset_plate_deltas_5sim20t.hdf5' #adjust dataset path
	standardization_method = 'std'

	plot_intermediate_fields = True
	save_plots = True
	show_plots = False
	apply_filter = False
	create_GIF = True

	n_sims = 5
	n_ts = 5

	call_SM_main(delta, model_name, shape, overlap_ratio, var_p, var_in, max_num_PC, dataset_path,	\
				plot_intermediate_fields, standardization_method, save_plots, show_plots, apply_filter, create_GIF, \
				n_sims, n_ts)

