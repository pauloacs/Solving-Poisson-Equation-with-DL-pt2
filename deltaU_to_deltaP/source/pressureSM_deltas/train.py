from ctypes import py_object
from re import X
import matplotlib
from numpy.core.defchararray import array

import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import inplace_update
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import dask
import dask.config
import dask.distributed
import dask_ml
import dask_ml.preprocessing
import dask_ml.decomposition

from pyDOE import lhs
from numba import njit
import tensorflow as tf
import os
import shutil
import time
import h5py
import numpy as np
import tensorflow as tf
tf.keras.utils.set_random_seed(0)

from tensorflow.keras import Model
from tensorflow.keras.layers import ZeroPadding2D, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, concatenate, Input
import math
import scipy.spatial.qhull as qhull
import itertools
from scipy.spatial import cKDTree as KDTree

import matplotlib.pyplot as plt
from scipy.spatial import distance
import matplotlib.path as mpltPath
from shapely.geometry import MultiPoint
from sklearn.decomposition import PCA, IncrementalPCA
import tables
import pickle as pk


class Training:

  def __init__(self, delta, block_size, var_p, var_in, hdf5_paths, Nsamples, numSims, numTimeFrames, standardization_method):
    self.delta = delta
    self.block_size = block_size
    self.var_in = var_in
    self.var_p = var_p
    self.paths = hdf5_paths
    self.Nsamples = Nsamples
    self.numSims = numSims
    self.numTimeFrames = numTimeFrames
    self.standardization_method = standardization_method
  #@njit     ### with numba.njit is much faster but is returning an error when using it within the class ###
  def index(self, array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    # If no item was found return None, other return types might be a problem due to
    # numbas type inference.

  def interp_weights(self, xyz, uvw):
    d = 2 #2d interpolation
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

  def interpolate(self, values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

  def interpolate_fill(self, values, vtx, wts, fill_value=np.nan):  #this would be the function to fill with nan 
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)  #does not work yet
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret
  
  def create_uniform_grid(self, x_min, x_max,y_min,y_max): 
    """
    Creates a uniform quadrangular grid envolving every cell of the mesh
    """
    X0 = np.linspace(x_min + self.delta/2 , x_max - self.delta/2 , num = int(round( (x_max - x_min)/self.delta )) )
    Y0 = np.linspace(y_min + self.delta/2 , y_max - self.delta/2 , num = int(round( (y_max - y_min)/self.delta )) )

    XX0, YY0 = np.meshgrid(X0,Y0)
    return XX0.flatten(), YY0.flatten()


  def densePCA(self, n_layers, depth=512, dropout_rate=False):
    """
    Creates the MLP NN.
    """

    inputs = Input(self.PC_input)
    if len(depth) == 1:
      depth = [depth]*n_layers
    x = tf.keras.layers.Dense(depth[0], activation='relu')(inputs)
    if dropout_rate: x = tf.keras.layers.Dropout(0.2)(x)
    for i in range(n_layers - 1):
      x = tf.keras.layers.Dense(depth[i+1], activation='relu')(x)
      if dropout_rate: x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(self.PC_p)(x)

    model = Model(inputs, outputs, name="MLP")
    print(model.summary())

    return model


  def domain_dist(self, i,top_boundary, obst_boundary, xy0):

    # boundaries indice
    indice_top = self.index(top_boundary[i,0,:,0] , -100.0 )[0]
    top = top_boundary[i,0,:indice_top,:]
    max_x, max_y, min_x, min_y = np.max(top[:,0]), np.max(top[:,1]) , np.min(top[:,0]) , np.min(top[:,1])


    #extra_points = np.array( [ ( np.min(top[:,0]) , 0 ), ( np.max(top[:,0]), 0 ) ] )  #to generalize just use top+inlet+ outlet and works for any domain and just remove this
    #top = np.concatenate((top,extra_points), axis=0)  #np.c_[ np.append(top[:,0], extra_points) , np.append(top[:,1], extra_points) ]   

    is_inside_domain = ( xy0[:,0] <= max_x)  * ( xy0[:,0] >= min_x ) * ( xy0[:,1] <= max_y ) * ( xy0[:,1] >= min_y ) #this is just for simplification

    # regular polygon for testing

    # # Matplotlib mplPath
    # path = mpltPath.Path(top_inlet_outlet)
    # is_inside_domain = path.contains_points(xy0)
    # print(is_inside_domain.shape)

    indice_obst = self.index(obst_boundary[i,0,:,0] , -100.0 )[0]
    obst = obst_boundary[i,0,:indice_obst,:]

    obst_points =  MultiPoint(obst)

    hull = obst_points.convex_hull       #only works for convex geometries
    hull_pts = hull.exterior.coords.xy    #have a code for any geometry . enven concave https://stackoverflow.com/questions/14263284/create-non-intersecting-polygon-passing-through-all-given-points/47410079
    hull_pts = np.c_[hull_pts[0], hull_pts[1]]

    path = mpltPath.Path(hull_pts)
    is_inside_obst = path.contains_points(xy0)

    domain_bool = is_inside_domain * ~is_inside_obst

    top = top[0:top.shape[0]:5,:]   #if this has too many values, using cdist can crash the memmory since it needs to evaluate the distance between ~1M points with thousands of points of top
    obst = obst[0:obst.shape[0]:5,:]

    print(top.shape)

    sdf = np.minimum( distance.cdist(xy0,obst).min(axis=1) , distance.cdist(xy0,top).min(axis=1) ) * domain_bool
    print(np.max(distance.cdist(xy0,top).min(axis=1)))
    print(np.max(sdf))

    return domain_bool, sdf

  def read_dataset(self):

      #pathCil, pathRect, pathTria , pathPlate = self.paths[0], self.paths[1], self.paths[2], self.paths[3]
      pathCil = self.paths[0]

      NUM_COLUMNS = 4

      file = tables.open_file(self.filename, mode='w')
      atom = tables.Float32Atom()

      array_c = file.create_earray(file.root, 'data', atom, (0, self.block_size, self.block_size, NUM_COLUMNS))
      file.close()

      max_abs_Ux = []
      max_abs_Uy = []
      max_abs_dist = []
      max_abs_p = []

      sim_cil = False
      sim_tria = False   
      sim_placa = False 

      hdf5_file = h5py.File(pathCil, "r")
      data = hdf5_file["sim_data"][:self.numSimsCil, :self.numTimeFrames, ...]
      top_boundary = hdf5_file["top_bound"][:self.numSimsCil, :self.numTimeFrames, ...]
      obst_boundary = hdf5_file["obst_bound"][:self.numSimsCil, :self.numTimeFrames,  ...]
      hdf5_file.close()

      for i in range(np.sum(self.numSims)):
          
          indice = self.index(data[i,0,:,0] , -100.0 )[0]
          data_limited = data[i,0,:indice,:]

          x_min = round(np.min(data_limited[...,3]),2)
          x_max = round(np.max(data_limited[...,3]),2)

          y_min = round(np.min(data_limited[...,4]),2) #- 0.1
          y_max = round(np.max(data_limited[...,4]),2) #+ 0.1


          ######### -------------------- Assuming constant mesh, the following can be done out of the for cycle ------------------------------- ##########

          X0, Y0 = self.create_uniform_grid(x_min, x_max, y_min, y_max)
          xy0 = np.concatenate((np.expand_dims(X0, axis=1),np.expand_dims(Y0, axis=1)), axis=-1)
          points = data_limited[...,3:5] #coordinates

          vert, weights = self.interp_weights(points, xy0) #takes ~100% of the interpolation cost and it's only done once for each different mesh/simulation case
      
          domain_bool, sdf = self.domain_dist(i, top_boundary, obst_boundary, xy0)

          div = 1 #parameter defining the sliding window vertical and horizontal displacements
          
          self.grid_shape_y = int(round((y_max-y_min)/self.delta)) 
          self.grid_shape_x = int(round((x_max-x_min)/self.delta)) 

          count_ = data.shape[1]* int(self.grid_shape_y/div - self.block_size/div + 1 ) * int(self.grid_shape_x/div - self.block_size/div + 1 )

          num_samples = int(self.Nsamples)    #number of samples from each sim -------> needs to be adjusted

          count = 0
          cicle = 0

          #arrange data in array: #this can be put outside the j loop if the mesh is constant 

          x0 = np.min(X0)
          y0 = np.min(Y0)
          dx = self.delta
          dy = self.delta

          indices= np.zeros((X0.shape[0],2))
          obst_bool  = np.zeros((self.grid_shape_y,self.grid_shape_x,1))
          sdfunct = np.zeros((self.grid_shape_y,self.grid_shape_x,1))

          delta_p = data_limited[...,7:8] #values
          p_interp = self.interpolate_fill(delta_p, vert, weights) 
        
          for (step, x_y) in enumerate(xy0):  
              if domain_bool[step] * (~np.isnan(p_interp[step])) :
                  jj = int(round((x_y[...,0] - x0) / dx))
                  ii = int(round((x_y[...,1] - y0) / dy))

                  indices[step,0] = ii
                  indices[step,1] = jj
                  sdfunct[ii,jj,:] = sdf[step]
                  obst_bool[ii,jj,:]  = int(1)

          indices = indices.astype(int)

          for j in range(data.shape[1]):  #100 for both data and data_rect

            data_limited = data[i,j,:indice,:]#[mask_x]

            Ux = data_limited[...,0:1] #values
            Uy = data_limited[...,1:2] #values

            delta_p = data_limited[...,7:8] #values
            delta_Ux = data_limited[...,5:6] #values
            delta_Uy = data_limited[...,6:7] #values

            U_max_norm = np.max(np.sqrt(np.square(Ux) + np.square(Uy)))

            delta_p_adim = delta_p/pow(U_max_norm,2.0) 
            delta_Ux_adim = delta_Ux/U_max_norm 
            delta_Uy_adim = delta_Uy/U_max_norm 

            delta_p_interp = self.interpolate_fill(delta_p_adim, vert, weights) #compared to the griddata interpolation 
            delta_Ux_interp = self.interpolate_fill(delta_Ux_adim, vert, weights)#takes virtually no time  because "vert" and "weigths" where already calculated
            delta_Uy_interp = self.interpolate_fill(delta_Uy_adim, vert, weights)

            #arrange data in array:

            grid = np.zeros(shape=(1, self.grid_shape_y, self.grid_shape_x, 4))

            grid[0,:,:,0:1][tuple(indices.T)] = delta_Ux_interp.reshape(delta_Ux_interp.shape[0],1)
            grid[0,:,:,1:2][tuple(indices.T)] = delta_Uy_interp.reshape(delta_Uy_interp.shape[0],1)
            grid[0,:,:,2:3] = sdfunct
            grid[0,:,:,3:4][tuple(indices.T)] = delta_p_interp.reshape(delta_p_interp.shape[0],1)

            x_list = []
            obst_list = []
            y_list = []

            grid[np.isnan(grid)] = 0 #set any nan value to 0

            lb = np.array([0 + self.block_size * self.delta/2 , 0 + self.block_size * self.delta/2 ])
            ub = np.array([(x_max-x_min) - self.block_size * self.delta/2, (y_max-y_min) - self.block_size * self.delta/2])

            XY = lb + (ub-lb)*lhs(2,int(num_samples/self.numTimeFrames))  #divided by 100 because it samples from each time individually
            XY_indices = (np.round(XY/self.delta)).astype(int)

            new_XY_indices = [tuple(row) for row in XY_indices]
            XY_indices = np.unique(new_XY_indices, axis=0)

            for [jj, ii] in XY_indices:

                    i_range = [int(ii - self.block_size/2), int( ii + self.block_size/2) ]
                    j_range = [int(jj - self.block_size/2), int( jj + self.block_size/2) ]

                    x_u = grid[0, i_range[0]:i_range[1] , j_range[0]:j_range[1] , 0:2 ]
                    x_obst = grid[0, i_range[0]:i_range[1] , j_range[0]:j_range[1] , 2:3 ]
                    y = grid[0, i_range[0]:i_range[1] , j_range[0]:j_range[1] , 3:4 ]

                    # Remove all the blocks with delta_U = 0 and delta_p = 0
                    if not ((x_u == 0).all() and (y==0).all()):
                      x_list.append(x_u)
                      #x_list.append(grid[0, i:(i + 100), j:(j + 100), 0:2 ].transpose(1,0,2))
                      
                      obst_list.append(x_obst)
                      # #obst_list.append(grid[0, i:(i + 100), j:(j + 100), 2:3 ].transpose(1,0,2))

                      y_list.append(y)
                      # #y_list.append(grid[0, i:(i + 100), j:(j + 100), 3:4 ].transpose(1,0,2))

            cicle += 1
            print(cicle, flush = True)

            x_array = np.array(x_list, dtype = 'float16')#, axis=0 )#.astype('float32') #, dtype = 'float16')
            obst_array = np.array(obst_list, dtype = 'float16')#, axis=0 )#.astype('float16') #np.array(obst_list, dtype = 'float16')
            y_array = np.array(y_list, dtype = 'float16')#, axis=0 )#.astype('float16') #np.array(y_list, dtype = 'float16')

            max_abs_Ux.append(np.max(np.abs(x_array[...,0])))
            max_abs_Uy.append(np.max(np.abs(x_array[...,1])))
            max_abs_dist.append(np.max(np.abs(obst_array[...,0])))

            for step in range(y_array.shape[0]):
              y_array[step,...][obst_array[step,...] != 0] -= np.mean(y_array[step,...][obst_array[step,...] != 0])
            
            max_abs_p.append(np.max(np.abs(y_array[...,0])))
            
            array = np.c_[x_array,obst_array,y_array]
            
            ### removing duplicate data

            reshaped_array = array.reshape(array.shape[0], -1)

            # Find unique rows
            unique_indices = np.unique(reshaped_array, axis=0, return_index=True)[1]

            # Select unique rows
            unique_array = array[unique_indices]

            # Split the unique array back into individual arrays
            x_unique_array = unique_array[:, :, :, :2]
            obst_unique_array = unique_array[:, :, :, 2:3]
            y_unique_array = unique_array[:, :, :, 3:]
            
            file = tables.open_file(self.filename, mode='a')
            file.root.data.append(array)
            file.close()

      self.max_abs_Ux = np.max(np.abs(max_abs_Ux))
      self.max_abs_Uy = np.max(np.abs(max_abs_Uy))
      self.max_abs_dist = np.max(np.abs(max_abs_dist))
      self.max_abs_p = np.max(np.abs(max_abs_p))

      np.savetxt('maxs', [self.max_abs_Ux, self.max_abs_Uy, self.max_abs_dist, self.max_abs_p] )
          
      return 0

  #coding the training

  @tf.function
  def train_step(self, inputs, labels):
    with tf.GradientTape() as tape:
      predictions = self.model(inputs, training=True)
      loss=self.loss_object(labels, predictions)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    return loss

  #@tf.function
  def perform_validation(self):

    losses = []

    for (x_val, y_val) in self.test_dataset:
      x_val = tf.cast(x_val[...,0,0], dtype='float32')
      y_val = tf.cast(y_val[...,0,0], dtype='float32')

      val_logits = self.model(x_val)
      val_loss = self.loss_object(y_true = y_val , y_pred = val_logits)
      losses.append(val_loss)

    return losses
  
  def my_mse_loss(self):
    def loss_f(y_true, y_pred):

      loss = tf.reduce_mean(tf.square(y_true - y_pred) )  #weighted loss -  more relante PC weigth more 

      return   1e6 * loss
    return loss_f

  def _bytes_feature(self, value):
      """Returns a bytes_list from a string / byte."""
      if isinstance(value, type(tf.constant(0))):
          value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  def _float_feature(self, value):
      """Returns a float_list from a float / double."""
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
  def _int64_feature(self, value):
      """Returns an int64_list from a bool / enum / int / uint."""
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def parse_single_image(self, input_parse, output_parse):

    #define the dictionary -- the structure -- of our single example
    data = {
    'height' : self._int64_feature(input_parse.shape[0]),
          'depth_x' : self._int64_feature(input_parse.shape[1]),
          'depth_y' : self._int64_feature(output_parse.shape[1]),
          'raw_input' : self._bytes_feature(tf.io.serialize_tensor(input_parse)),
          'output' : self._bytes_feature(tf.io.serialize_tensor(output_parse)),
      }

    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

  def write_images_to_tfr_short(self, input, output, filename:str="images"):
    filename= filename+".tfrecords"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0

    for index in range(len(input)):

      #get the data we want to write
      current_input = input[index].astype('float64')
      current_output = output[index].astype('float64')

      out = self.parse_single_image(input_parse=current_input, output_parse=current_output)
      writer.write(out.SerializeToString())
      count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count

  
  def unison_shuffled_copies(self, a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
  
  def apply_PCA(self, filename_flat, max_num_PC):

    file = tables.open_file(filename_flat, mode='w')
    atom = tables.Float32Atom()

    file.create_earray(file.root, 'data_flat', atom, (0, max_num_PC, 2))
    file.close()

    client = dask.distributed.Client(processes=False)#, n_workers=16)

    self.ipca_p = dask_ml.decomposition.IncrementalPCA(max_num_PC)
    self.ipca_input = dask_ml.decomposition.IncrementalPCA(max_num_PC)

    N = int(self.Nsamples * (self.numSimsCil)) # +self.numSimsRect + self.numSimsTria + self.numSimsPlate))

    chunk_size = int(N/10)
    print('Passing the PCA ' + str(N//chunk_size) + ' times', flush = True)

    for i in range(int(N//chunk_size)):

      f = tables.open_file(self.filename, mode='r')
      x_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,0:2] # e.g. read from disk only this par$
      obst_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,2:3]
      y_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,3:4]
      f.close()

      if x_array.shape[0] < max_num_PC:
        print('This chunck is too small ... skipping')
        break

      x_array_flat = x_array.reshape((x_array.shape[0], x_array.shape[1]*x_array.shape[2], 2 ))
      x_array_flat1 = x_array_flat[...,0:1]/self.max_abs_Ux
      x_array_flat2 = x_array_flat[...,1:2]/self.max_abs_Uy
      obst_array_flat = obst_array.reshape((obst_array.shape[0], obst_array.shape[1]*obst_array.shape[2], 1 ))/self.max_abs_dist

      y_array_flat = y_array.reshape((y_array.shape[0], y_array.shape[1]*y_array.shape[2]))/self.max_abs_p

      input_flat = np.concatenate((x_array_flat1,x_array_flat2,obst_array_flat) , axis = -1)
      input_flat = input_flat.reshape((input_flat.shape[0],-1))
      y_flat = y_array_flat.reshape((y_array_flat.shape[0],-1)) 

      input_dask = dask.array.from_array(input_flat, chunks='auto')
      y_dask = dask.array.from_array(y_array_flat, chunks='auto')

      #scaler = dask_ml.preprocessing.StandardScaler().fit(input_dask)
      scaler = dask_ml.preprocessing.StandardScaler().fit(input_dask.rechunk({1: input_dask.shape[1]}))

      #scaler1 = dask_ml.preprocessing.StandardScaler().fit(y_dask)
      scaler1 = dask_ml.preprocessing.StandardScaler().fit(y_dask.rechunk({1: y_dask.shape[1]}))

      inputScaled = scaler.transform(input_dask)
      yScaled = scaler1.transform(y_dask)

      self.ipca_input.partial_fit(inputScaled)
      self.ipca_p.partial_fit(yScaled)

      print('Fitted ' + str(i+1) + '/' + str(N//chunk_size), flush = True)

    self.PC_p = np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) if np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) > 1 and np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) <= max_num_PC else max_num_PC  #max defined to be 32 here
    self.PC_input = np.argmax(self.ipca_input.explained_variance_ratio_.cumsum() > self.var_in) if np.argmax(self.ipca_input.explained_variance_ratio_.cumsum() > self.var_in) > 1 and np.argmax(self.ipca_input.explained_variance_ratio_.cumsum() > self.var_in) <= max_num_PC else max_num_PC

    print('PC_p :' + str(self.PC_p), flush = True)
    print('PC_input :' + str(self.PC_input), flush = True)

    print(' Total variance from input represented: ' + str(np.sum(self.ipca_input.explained_variance_ratio_[:self.PC_input])))
    pk.dump(self.ipca_input, open("ipca_input.pkl","wb"))

    print(' Total variance from p represented: ' + str(np.sum(self.ipca_p.explained_variance_ratio_[:self.PC_p])))
    pk.dump(self.ipca_p, open("ipca_p.pkl","wb"))

    for i in range(int(N//chunk_size)):

      f = tables.open_file(self.filename, mode='r')
      x_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,0:2] # e.g. read from disk only this part of the dataset
      obst_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,2:3]
      y_array = f.root.data[i*chunk_size:(i+1)*chunk_size,:,:,3:4]
      f.close()

      x_array_flat = x_array.reshape((x_array.shape[0], x_array.shape[1]*x_array.shape[2], 2 ))
      x_array_flat1 = x_array_flat[...,0:1]/self.max_abs_Ux
      x_array_flat2 = x_array_flat[...,1:2]/self.max_abs_Uy
      obst_array_flat = obst_array.reshape((obst_array.shape[0], obst_array.shape[1]*obst_array.shape[2], 1 ))/self.max_abs_dist
      input_flat = np.concatenate((x_array_flat1,x_array_flat2,obst_array_flat) , axis = -1)
      input_flat = input_flat.reshape((input_flat.shape[0],-1))

      y_array_flat = y_array.reshape((y_array.shape[0], y_array.shape[1]*y_array.shape[2]))/self.max_abs_p

      input_dask = dask.array.from_array(input_flat, chunks='auto')
      y_dask = dask.array.from_array(y_array_flat, chunks='auto')

      scaler = dask_ml.preprocessing.StandardScaler().fit(input_dask.rechunk({1: input_dask.shape[1]}))
      scaler1 = dask_ml.preprocessing.StandardScaler().fit(y_dask.rechunk({1: y_dask.shape[1]}))

      inputScaled = scaler.transform(input_dask)
      yScaled = scaler1.transform(y_dask)

      input_transf = self.ipca_input.transform(input_flat)#[:,:self.PC_input]
      y_transf = self.ipca_p.transform(y_array_flat)#[:,:self.PC_input]

      array_image = np.concatenate((np.expand_dims(input_transf, axis=-1) , np.expand_dims(y_transf, axis=-1)), axis = -1)#, y_array]
      print(array_image.shape, flush = True)

      f = tables.open_file(filename_flat, mode='a')
      f.root.data_flat.append(array_image)
      f.close()
      
      print('transformed ' + str(i+1) + '/' + str(N//chunk_size), flush = True)


    client.close()
    
  def prepare_data (self, hdf5_paths, max_num_PC, outarray_fn = 'outarray.h5'):

    #load data

    print_shape = True

    #self.numSimsCil, self.numSimsRect, self.numSimsTria , self.numSimsPlate = self.numSims[0], self.numSims[1], self.numSims[2], self.numSims[3] 
    self.numSimsCil = self.numSims[0]
    self.filename = outarray_fn
      
    if not (os.path.isfile(outarray_fn) and os.path.isfile('maxs')):
      self.read_dataset()
    else:
      maxs = np.loadtxt('maxs')
      self.max_abs_Ux, self.max_abs_Uy, self.max_abs_dist, self.max_abs_p = maxs[0], maxs[1], maxs[2], maxs[3]
      
    filename_flat = 'outarray_flat.h5'

    if not (os.path.isfile(filename_flat) and os.path.isfile('ipca_input.pkl') and os.path.isfile('ipca_p.pkl')):
      print('Applying PCA \n')
      self.apply_PCA(filename_flat, max_num_PC)
    else:
      print('Data after PCA is available, load it and stepping over the PC analysis \n')
      self.ipca_p = pk.load(open("ipca_p.pkl",'rb'))
      self.ipca_input = pk.load(open("ipca_input.pkl",'rb'))

      self.PC_p = np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) if np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) > 1 and np.argmax(self.ipca_p.explained_variance_ratio_.cumsum() > self.var_p) <= max_num_PC else max_num_PC  #max defined to be 32 here
      self.PC_input = int(np.argmax(self.ipca_input.explained_variance_ratio_.cumsum() > self.var_in))

    f = tables.open_file(filename_flat, mode='r')
    input = f.root.data_flat[...,:self.PC_input,0] 
    output = f.root.data_flat[...,:self.PC_p,1] 
    f.close()

    # Treating PCA data

    if self.standardization_method == 'min_max':
      ## Option 2: Min-max scaling
      min_in = np.min(input, axis=0)
      max_in = np.max(input, axis=0)

      min_out = np.min(output, axis=0)
      max_out = np.max(output, axis=0)

      np.savez('min_max_values.npz', min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out)

      # Perform min-max scaling
      x = (input - min_in) / (max_in - min_in)
      y = (output - min_out) / (max_out - min_out)
      
    elif self.standardization_method == 'std':
      ## Option 1: Standardization
      mean_in = np.mean(input, axis=0)
      std_in = np.std(input, axis=0)

      mean_out = np.mean(output, axis=0)
      std_out = np.std(output, axis=0)

      np.savez('mean_std.npz', mean_in=mean_in, std_in=std_in, mean_out=mean_out, std_out=std_out)

      x = (input - mean_in) /std_in
      y = (output - mean_out) /std_out
    elif self.standardization_method == 'max_abs':

      # Option 3 - Old method
      max_abs_input_PCA = np.max(np.abs(input))
      max_abs_p_PCA = np.max(np.abs(output))
      print( max_abs_input_PCA, max_abs_p_PCA)

      np.savetxt('maxs_PCA', [max_abs_input_PCA, max_abs_p_PCA] )

      x = input/max_abs_input_PCA
      y = output/max_abs_p_PCA

    x, y = self.unison_shuffled_copies(x, y)
    print('Data shuffled \n')

    x = x.reshape((x.shape[0], x.shape[1], 1, 1))
    y = y.reshape((y.shape[0], y.shape[1], 1, 1))

    #tf records

    # Convert values to compatible tf.Example types.

    split = 0.9
    if not (os.path.isfile('train_data.tfrecords') and os.path.isfile('test_data.tfrecords')):
      count = self.write_images_to_tfr_short(x[:int(split*x.shape[0]),...], y[:int(split*y.shape[0]),...], filename="train_data")
      count = self.write_images_to_tfr_short(x[int(split*x.shape[0]):,...], y[int(split*y.shape[0]):,...], filename="test_data")
    else:
      print("TFRecords train and test data already available, using it... If you want to write new data, delete 'train_data.tfrecords' and 'test_data.tfrecords'!")
    self.len_train = int(split*x.shape[0])

    return 0 

  def parse_tfr_element(self, element):
    #use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'output' : tf.io.FixedLenFeature([], tf.string),
        'raw_input' : tf.io.FixedLenFeature([], tf.string),
        'depth_x':tf.io.FixedLenFeature([], tf.int64),
        'depth_y':tf.io.FixedLenFeature([], tf.int64)
      }

    content = tf.io.parse_single_example(element, data)
    
    height = content['height']
    depth_x = content['depth_x']
    depth_y = content['depth_y']
    output = content['output']
    raw_input = content['raw_input']
    
    
    #get our 'feature'-- our image -- and reshape it appropriately
    
    input_out= tf.io.parse_tensor(raw_input, out_type=tf.float64)
    output_out = tf.io.parse_tensor(output, out_type=tf.float64)

    return ( input_out , output_out)

  def load_dataset(self, filename, batch_size, buffer_size):
    #create the dataset
    dataset = tf.data.TFRecordDataset(filename)

      #pass every single feature through our mapping function
    dataset = dataset.map(
        self.parse_tfr_element
    )

    dataset = dataset.shuffle(buffer_size=buffer_size )
    #epoch = tf.data.Dataset.range(epoch_num)
    dataset = dataset.batch(batch_size)

    return dataset  
    
  def Callback_EarlyStopping(self, LossList, min_delta=0.1, patience=20):
      #No early stopping for 2*patience epochs
      if len(LossList)//patience < 2 :
          return False
      #Mean loss for last patience epochs and second-last patience epochs
      mean_previous = np.mean(LossList[::-1][patience:2*patience]) #second-last
      mean_recent = np.mean(LossList[::-1][:patience]) #last
      #you can use relative or absolute change
      delta_abs = np.abs(mean_recent - mean_previous) #abs change
      delta_abs = np.abs(delta_abs / mean_previous)  # relative change
      if delta_abs < min_delta :
          print("*CB_ES* Loss didn't change much from last %d epochs"%(patience))
          print("*CB_ES* Percent change in loss value:", delta_abs*1e2)
          return True
      else:
          return False


  def load_data_And_train(self, lr, batch_size, max_num_PC, model_name, beta_1, num_epoch, n_layers, width, dropout_rate):

    train_path = 'train_data.tfrecords'
    test_path = 'test_data.tfrecords'

    self.train_dataset = self.load_dataset(filename = train_path, batch_size= batch_size, buffer_size=1024)
    self.test_dataset = self.load_dataset(filename = test_path, batch_size= batch_size, buffer_size=1024)

    # Training 

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=0.999, epsilon=1e-08)#, decay=0.45*lr, amsgrad=True)
    self.loss_object = self.my_mse_loss()

    #model = tf.keras.models.load_model('model_first_.h5') # to load model
    self.model = self.densePCA(n_layers, width, dropout_rate)

    epochs_val_losses, epochs_train_losses = [], []

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta_1, beta_2=0.999)#, epsilon=1e-08, decay=0.45*lr, amsgrad=True)

    for epoch in range(num_epoch):
      progbar = tf.keras.utils.Progbar(math.ceil(self.len_train/batch_size))
      print('Start of epoch %d' %(epoch,))
      losses_train = []
      losses_test = []

      for step, (inputs, labels) in enumerate(self.train_dataset):

        inputs = tf.cast(inputs[...,0,0], dtype='float32')
        labels = tf.cast(labels[...,0,0], dtype='float32')
        loss = self.train_step(inputs, labels)
        losses_train.append(loss)

      losses_val  = self.perform_validation()

      losses_train_mean = np.mean(losses_train)
      losses_val_mean = np.mean(losses_val)

      epochs_train_losses.append(losses_train_mean)
      epochs_val_losses.append(losses_val_mean)
      print('Epoch %s: Train loss: %.4f , Validation Loss: %.4f \n' % (epoch,float(losses_train_mean), float(losses_val_mean)))

      progbar.update(step+1)

      stopEarly = self.Callback_EarlyStopping(epochs_val_losses, min_delta=0.1/100, patience=250)
      if stopEarly:
        print("Callback_EarlyStopping signal received at epoch= %d/%d"%(epoch,num_epoch))
        break

      if epoch > 20:
        min_yet = losses_val_mean
        print('saving model')
        mod = 'model_' + model_name + '.h5'
        self.model.save(mod)
    
    print("Terminating training")
    mod = 'model_' + model_name + '.h5'
    ## Plot loss vs epoch
    plt.plot(list(range(len(epochs_train_losses))), epochs_train_losses, label ='train')
    plt.plot(list(range(len(epochs_val_losses))), epochs_val_losses, label ='val')
    plt.yscale('log')
    plt.legend()
    plt.savefig('loss_vs_epoch.png')

    ## Save losses data
    np.savetxt('train_loss' + str(beta_1)+ str(lr)+ '.txt', epochs_train_losses, fmt='%d')
    np.savetxt('test_loss' + str(beta_1)+ str(lr)+ '.txt', epochs_val_losses, fmt='%d')
        
    return 0
  
def main_train(dataset_path, num_sims, num_ts, num_epoch, lr, beta, batch_size, standardization_method, \
              n_samples, block_size, delta, max_num_PC, var_p, var_in, model_size, dropout_rate, outarray_fn):

  if model_size == 'small':
    n_layers = 3
    width = [512]*3
  elif model_size == 'medium':
    n_layers = 5
    width = [256] + [512]*3 + [256]
  elif model_size == 'big':
    n_layers = 8
    width = [256] + [512]*6 + [256]
  elif model_size == 'huge':
    n_layers = 12
    width = [256] + [512]*10 + [256]
  else:
    raise ValueError('Invalid NN model type')

  paths = [dataset_path]
  num_ts = [num_ts]
  num_sims = [num_sims]

  model_name = f'{model_size}-{model_size}-{standardization_method}-{var_p}-drop{dropout_rate}'

  Train = Training(delta, block_size,var_p, var_in, paths, n_samples, num_sims, num_ts, standardization_method)

  # If you want to read the crude dataset (hdf5) again, delete the 'outarray.h5' file
  Train.prepare_data (paths, max_num_PC, outarray_fn) #prepare and save data to tf records
  Train.load_data_And_train(lr, batch_size, max_num_PC, model_name, beta, num_epoch, n_layers, width, dropout_rate)

if __name__ == '__main__':

  path_placa = 'dataset_plate_deltas_5sim20t.hdf5'
  dataset_path = [path_placa]

  num_sims_placa = 5
  num_ts = [num_sims_placa]#, num_sims_rect, num_sims_tria, num_sims_placa]
  num_ts = 5

  # Training Parameters
  num_epoch = 5000
  lr = 1e-5
  beta = 0.99
  batch_size = 1024 #*8
  ## Possible methods:
  ## 'std', 'min_max' or 'max_abs'
  standardization_method = 'std'

  # Data-Processing Parameters
  n_samples = int(1e4) #no. of samples per simulation
  block_size = 128
  delta = 5e-3
  max_num_PC = 512 # to not exceed the width of the NN
  var_p = 0.95
  var_in = 0.95

  model_size = 'small'
  dropout_rate = 0.2

  outarray_fn = '../blocks_dataset/outarray.h5'

  main_train(dataset_path, num_sims, num_ts, num_epoch, lr, beta, batch_size, standardization_method, \
    n_samples, block_size, delta, max_num_PC, var_p, var_in, model_size, dropout_rate, outarray_fn)
