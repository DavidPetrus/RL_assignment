import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipdb
from colour_segmentation import colour_segmentation

img = plt.imread(r'asd.png')

img = img[:,:,:3]


asd = colour_segmentation(img, 'globe', 1)



ipdb.set_trace()
