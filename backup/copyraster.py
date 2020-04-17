import os
import shutil

for dir in filter(os.path.isdir, os.listdir(os.getcwd())):
 if os.path.isfile(dir +'/data/raster_plot.png'): 
  shutil.copy(dir +'/data/raster_plot.png', os.curdir)
  os.rename('raster_plot.png', 'raster_plot_' + dir + '.png')

 if os.path.isfile(dir +'/data/box_plot.png'): 
  shutil.copy(dir +'/data/box_plot.png', os.curdir)
  os.rename('box_plot.png', 'box_plot_' + dir + '.png')

 if os.path.isfile(dir +'/data/fr_hist_1ms_bin.png'):
  shutil.copy(dir + '/data/fr_hist_1ms_bin.png', os.curdir)
  os.rename('fr_hist_1ms_bin.png', 'fr_hist_1ms_bin_' + dir + '.png')

 if os.path.isfile(dir +'/data/fr_hist_0.1ms_bin.png'):
  shutil.copy(dir + '/data/fr_hist_0.1ms_bin.png', os.curdir)
  os.rename('fr_hist_0.1ms_bin.png', 'fr_hist_0.1ms_bin_' + dir + '.png')
 
 
 
