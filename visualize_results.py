import caffe
import matplotlib.pylab as plt
import numpy as np
import h5py
import copy


def vis_square_color_pts(mydata,pts,timesteps,filterind=[],redpoints=[], padsize=1, padval=0,alpha=1.0, filename=[]):
  ind = timesteps[0]
  if len(timesteps) == 1:
    data = copy.deepcopy(mydata[ind:ind+1,:,:,:].transpose(0,2,3,1))
  else:
    if len(filterind) > 0:
      print "WARNING: points may not be correct"
    data = copy.deepcopy(mydata[np.sort(timesteps),:,:,:].transpose(0,2,3,1))
  #for i in range(data.shape[0]):
  #  data[i,:,:,:] -= data[i,:,:,:].max()
  #  data[i,:,:,:] /= data[i,:,:,:].min()
  data -= data.min()
  data /= data.max()

  img_width = mydata.shape[2]


  # force the number of filters to be square
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
  data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
  # tile the filters into an image
  data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
  data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

  plt.imshow(data, alpha = alpha)
  plt.axis("off")
  if len(filterind) != 0:
    ax=plt.gca();
    NUM_COLORS=len(filterind)
    cm = plt.get_cmap('nipy_spectral');
    redi = []
    #ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]);
    for i in range(len(filterind)):
        filteri = filterind[i]
        if filteri in redpoints:
          redi.append(filteri);
          plt.plot((pts[ind,filteri*2]+1)/2.0*img_width,(pts[ind,filteri*2+1]+1)/2.0*img_width,'or', markersize=10);
        else:
          #plt.plot((pts[ind,filteri*2]+1)/2.0*img_width,(pts[ind,filteri*2+1]+1)/2.0*img_width,'o', color=cm((1.*i)/NUM_COLORS),markersize=7);
          plt.plot((pts[ind,filteri*2]+1)/2.0*img_width,(pts[ind,filteri*2+1]+1)/2.0*img_width,'ob', markersize=10,markeredgecolor=cm((1.*i)/NUM_COLORS),fillstyle='none', markeredgewidth=3.0);
    for filteri in redi:
        plt.plot((pts[ind,filteri*2]+1)/2.0*img_width,(pts[ind,filteri*2+1]+1)/2.0*img_width,'or', markersize=10);
    plt.xlim([0,img_width]);plt.ylim([img_width,0]);
  if filename: plt.savefig(filename, bbox_inches='tight')

def vis_square_color_pts_compare(mydata,pts,pts2,timesteps,filterind=[],redpoints=[], padsize=1, padval=0,alpha=1.0, filename=[]):
  ind = timesteps[0]
  if len(timesteps) == 1:
    data = copy.deepcopy(mydata[ind:ind+1,:,:,:].transpose(0,2,3,1))
  else:
    if len(filterind) > 0:
      print "WARNING: points may not be correct"
    data = copy.deepcopy(mydata[np.sort(timesteps),:,:,:].transpose(0,2,3,1))
  #for i in range(data.shape[0]):
  #  data[i,:,:,:] -= data[i,:,:,:].max()
  #  data[i,:,:,:] /= data[i,:,:,:].min()
  data -= data.min()
  data /= data.max()

  img_width = mydata.shape[2]


  # force the number of filters to be square
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
  data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
  # tile the filters into an image
  data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
  data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

  plt.imshow(data, alpha = alpha)
  plt.axis("off")
  if len(filterind) != 0:
    ax=plt.gca();
    NUM_COLORS=len(filterind)
    cm = plt.get_cmap('nipy_spectral');
    redi = []
    #ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]);
    for i in range(len(filterind)):
        filteri = filterind[i]
        plt.plot((pts[ind,filteri*2]+1)/2.0*img_width,(pts[ind,filteri*2+1]+1)/2.0*img_width,'x', markersize=12,markeredgecolor=(.894,0,0,1.0),fillstyle='none', markeredgewidth=3.0);
        plt.plot((pts2[ind,filteri*2]+1)/2.0*img_width,(pts2[ind,filteri*2+1]+1)/2.0*img_width,'ob', markersize=12,markeredgecolor=(0,0,.894,0.7),fillstyle='none', markeredgewidth=3.0);
          #plt.plot((pts[ind,filteri*2]+1)/2.0*img_width,(pts[ind,filteri*2+1]+1)/2.0*img_width,'o', color=cm((1.*i)/NUM_COLORS),markersize=7);
          #plt.plot((pts[ind,filteri*2]+1)/2.0*img_width,(pts[ind,filteri*2+1]+1)/2.0*img_width,'ob', markersize=10);
    for filteri in redi:
        plt.plot((pts[ind,filteri*2]+1)/2.0*img_width,(pts[ind,filteri*2+1]+1)/2.0*img_width,'or', markersize=10);
    plt.xlim([0,img_width]);plt.ylim([img_width,0]);
  if filename: plt.savefig(filename, bbox_inches='tight')

def vis_square_color_pts_compare_discrep(mydata,pts,pts2,timesteps,timesteps2,filterind=[],redpoints=[], padsize=1, padval=0,alpha=1.0, filename=[]):
  ind = timesteps[0]
  ind2 = timesteps2[0]
  if len(timesteps) == 1:
    data = copy.deepcopy(mydata[ind:ind+1,:,:,:].transpose(0,2,3,1))
  else:
    if len(filterind) > 0:
      print "WARNING: points may not be correct"
    data = copy.deepcopy(mydata[np.sort(timesteps),:,:,:].transpose(0,2,3,1))
  #for i in range(data.shape[0]):
  #  data[i,:,:,:] -= data[i,:,:,:].max()
  #  data[i,:,:,:] /= data[i,:,:,:].min()
  data -= data.min()
  data /= data.max()

  img_width = mydata.shape[2]


  # force the number of filters to be square
  n = int(np.ceil(np.sqrt(data.shape[0])))
  padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
  data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
  # tile the filters into an image
  data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
  data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

  plt.imshow(data, alpha = alpha)
  plt.axis("off")
  if len(filterind) != 0:
    ax=plt.gca();
    NUM_COLORS=len(filterind)
    cm = plt.get_cmap('nipy_spectral');
    redi = []
    #ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]);
    for i in range(len(filterind)):
        filteri = filterind[i]
        plt.plot((pts[ind,filteri*2]+1)/2.0*img_width,(pts[ind,filteri*2+1]+1)/2.0*img_width,'x', markersize=12,markeredgecolor=(.894,0,0,1.0),fillstyle='none', markeredgewidth=3.0);
        plt.plot((pts2[ind2,filteri*2]+1)/2.0*img_width,(pts2[ind2,filteri*2+1]+1)/2.0*img_width,'ob', markersize=12,markeredgecolor=(0,0,.894,0.7),fillstyle='none', markeredgewidth=3.0);
          #plt.plot((pts[ind,filteri*2]+1)/2.0*img_width,(pts[ind,filteri*2+1]+1)/2.0*img_width,'o', color=cm((1.*i)/NUM_COLORS),markersize=7);
          #plt.plot((pts[ind,filteri*2]+1)/2.0*img_width,(pts[ind,filteri*2+1]+1)/2.0*img_width,'ob', markersize=10);
    for filteri in redi:
        plt.plot((pts[ind,filteri*2]+1)/2.0*img_width,(pts[ind,filteri*2+1]+1)/2.0*img_width,'or', markersize=10);
    plt.xlim([0,img_width]);plt.ylim([img_width,0]);
  if filename: plt.savefig(filename, bbox_inches='tight')



def vis_square_pts(mydata,pts,timesteps,filterinds,pointinds=[], padsize=1, padval=1, filename=[]):
    if len(timesteps) == 1:
      assert(len(timesteps) == 1)
      t = timesteps[0];
      data = -copy.deepcopy(mydata[t:t+1,filterinds,:,:].transpose(1,2,3,0))
    else:
      assert(len(filterinds) == 1)
      i = filterinds[0]
      data = -copy.deepcopy(mydata[timesteps,i:i+1,:,:].transpose(0,2,3,1))
    for i in range(data.shape[0]):
      data[i,:,:,:] -= data[i,:,:,:].min()
      data[i,:,:,:] /= data[i,:,:,:].max()
    # data -= data.min()
    # data /= data.max()

    img_width = mydata.shape[2]

    print data.max()
    print data.min()
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data[:,:,0], cmap='Greys',  interpolation='nearest')
    plt.axis("off")
    if len(pointinds) != 0:
      assert(len(timesteps) == 1)
      t = timesteps[0];
      #ax=plt.gca();
      #NUM_COLORS=len(pointinds)
      #cm = plt.get_cmap('nipy_spectral');
      #ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]);
      for i in range(len(pointinds)):
          filteri = pointinds[i]
          plt.plot((pts[t,filteri*2]+1)/2.0*img_width,(pts[t,filteri*2+1]+1)/2.0*img_width,'or');
      plt.xlim([0,img_width]);plt.ylim([img_width,0]);
    if filename: plt.savefig(filename, bbox_inches='tight')

class PoseResults:
  def __init__(self, suffix):
    plt.rcParams['figure.figsize'] = (10, 10)
    prefix = "test_output/"
    #prefix = "pose_baseline/"
    #pose_output = h5py.File(prefix + suffix + ".h5", "r");
    #self.pose = pose_output['data']
    #self.pose_gt = pose_output['label']

    softmax_output = h5py.File(prefix +"softmax_" + suffix + ".h5", "r");
    self.softmax = softmax_output['data']
    self.points = softmax_output['label']

    conv1_output = h5py.File(prefix +"conv1_"+ suffix + ".h5", "r");
    conv2_output = h5py.File(prefix +"conv2_"+ suffix + ".h5", "r");
    conv3_output = h5py.File(prefix +"conv3_"+ suffix + ".h5", "r");
    self.conv1 = conv1_output['data']
    self.conv2 = conv2_output['data']
    self.conv3 = conv3_output['data']
    self.rgb = conv3_output['label']

    #self.get_mean_dist()
    #self.order_timesteps()

  # defines the distances, and sorts everything in decreasing order of accuracy
  def get_mean_dist(self, pt_idx=0):
    diff = np.asarray(self.pose)-np.asarray(self.pose_gt);
    diff = diff[:,pt_idx*3:pt_idx*3+3,0,0]
    diffsq = np.power(diff,2);
    eucsq = np.sum(diffsq,1);
    print 'network output: ', np.mean(eucsq)/2
    euc = np.power(eucsq,0.5);
    loss=np.mean(euc);
    std_dist = np.std(euc);
    print 'mean distance: ', loss
    print 'std distance: ', std_dist
    self.euc = euc;
  def order_timesteps(self):
    order = np.argsort(self.euc)
    self.euc = np.asarray([self.euc[i] for i in order])
    self.rgb = np.asarray([self.rgb[i,:,:,:] for i in order])
    self.conv1 = np.asarray([self.conv1[i,:,:,:] for i in order])
    self.conv2 = np.asarray([self.conv2[i,:,:,:] for i in order])
    self.conv3 = np.asarray([self.conv3[i,:,:,:] for i in order])
    self.softmax = np.asarray([self.softmax[i,:,:,:] for i in order])
    self.points = np.asarray([self.points[i,:,0,0] for i in order])

  # Returns mean and standard deviation in distance for each point and all points together.
  def get_mean_dists(self):
    all_eucs = []
    for pt_idx in range(3):
      diff = np.asarray(self.pose)-np.asarray(self.pose_gt);
      diff = diff[:,pt_idx*3:pt_idx*3+3,0,0]
      diffsq = np.power(diff,2);
      eucsq = np.sum(diffsq,1);
      print 'network output: ', np.mean(eucsq)/2
      euc = np.power(eucsq,0.5);
      all_eucs.extend(euc)
      loss=np.mean(euc);
      std_dist = np.std(euc);
      print 'mean distance: ', loss
      print 'std distance: ', std_dist
      self.euc = euc;

    std_dist = np.std(all_eucs);
    self.all_eucs = all_eucs
    print 'all mean = ', np.mean(all_eucs)
    print 'all std = ', np.std(all_eucs)


  def vis_rgb(self, timesteps = [], filters = [],padsize=1, padval=0):
    if timesteps == []:
      timesteps = [0];
    vis_square_color_pts(self.rgb, self.points, timesteps, filters,padsize=padsize, padval=padval);

  def vis_conv1(self, timesteps = [], filters = [],padsize=1, padval=0):
    if timesteps == []:
      timesteps = [0];
    if filters == []:
      filters = [0]
    vis_square_pts(self.conv1, self.points, timesteps, filters,padsize=padsize, padval=padval);

  def vis_conv2(self, timesteps = [], filters = [],padsize=1, padval=0):
    if timesteps == []:
      timesteps = [0];
    if filters == []:
      filters = [0]
    vis_square_pts(self.conv2, self.points, timesteps, filters,padsize=padsize, padval=padval);

  def vis_conv3(self, timesteps = [], filters = [],padsize=1, padval=0):
    if timesteps == []:
      timesteps = [0];
    if filters == []:
      filters = [0]
    vis_square_pts(self.conv3, self.points, timesteps, filters, padsize=padsize, padval=padval);

  def vis_softmax(self, timesteps = [], filters = [],padsize=1, padval=0):
    if timesteps == []:
      timesteps = [0];
    if filters == []:
      filters = [0]
    vis_square_pts(self.softmax, self.points, timesteps, filters,padsize=padsize, padval=padval);

  def vis_conv3_pts(self, timesteps = [], filters = [],padsize=1, padval=0):
    if timesteps == []:
      timesteps = [0];
    if filters == []:
      filters = [0]
    vis_square_pts(self.conv3, self.points, timesteps, filters, filters,padsize=padsize, padval=padval);

  def vis_softmax_pts(self, timesteps = [], filters = [],padsize=1, padval=0):
    if timesteps == []:
      timesteps = [0];
    if filters == []:
      filters = [0]
    vis_square_pts(self.softmax, self.points, timesteps, filters, filters,padsize=padsize, padval=padval);
