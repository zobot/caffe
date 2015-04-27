from scitools.std import movie

slow = PoseResults("slow")
for i in range(20):
	slow.vis_softmax_pts(timesteps = [i], filters = [0,1,2], pointinds = list(xrange(0,32)), filename = str(i) +'_0_all')

files = [str(x)+'_0_all.png' for x in list(xrange(0,20))]
movie(files,fps=2,output_file='feature_xy_points_all.gif')
