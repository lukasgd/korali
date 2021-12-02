import os, sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.colors as col
from mpl_toolkits.axes_grid1 import make_axes_locatable

class MidpointNormalize(col.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        col.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def purgeAllCsv(dir):
    csvList = os.listdir(dir)
    for filename in csvList:
        f = open(dir+filename, "w+")
        f.close()

os.mkdir('csvLabeled')
os.mkdir('csvOrderedY')
os.mkdir('csvOrderedFinal')
os.mkdir('p')
os.mkdir('u')
os.mkdir('v')


SCRATCH = "/scratch/snx3000/scatsoul/korali/"
SYSTEM = "Re100_largeStep/"
TEST_FOLDER = "_testingResults/"
SAMPLE = "sample000/"
PATH = SCRATCH+SYSTEM+TEST_FOLDER+SAMPLE

bpdx = 32
bpdy = 16
nAgents = bpdx*bpdy - 2*bpdx - 2*(bpdy-2)
x_res = bpdx - 2
y_res = bpdy - 2

p_cfd_tot, u_cfd_tot, v_cfd_tot, p_pred_tot, u_pred_tot, v_pred_tot, fileID_tot = [], [], [], [], [], [], []

# Purge existing files
purgeAllCsv('csvLabeled/')
purgeAllCsv('csvOrderedFinal/')
purgeAllCsv('csvOrderedY/')

# Order data and construct physical quantity field
def orderAndAssign(fname):

    fileID = os.path.basename(fname)[:4]

    # Read raw data and add header
    header_list = ["curStep","t","block_x","block_y","p_cfd","u_cfd","v_cfd","p_pred","u_pred","v_pred"]
    data = pd.read_csv(fname, header=None)
    data.to_csv("csvLabeled/" + "labeled" + os.path.basename(fname), header=header_list, index=False)

    # Read labeled data and sort y values
    labeledData = pd.read_csv("csvLabeled/" + "labeled" + os.path.basename(fname))
    labeledData.sort_values(["block_y"], axis=0, ascending=[True], inplace=True)
    labeledData.to_csv("csvOrderedY/" + "orderedY" + os.path.basename(fname), index=False)
    
    # Read y-ordered data
    orderedYdata = pd.read_csv("csvOrderedY/" + "orderedY" + os.path.basename(fname))

    # Sort x values: sequential read leads to start from bottom left
    ind = 0
    for i in range(int(nAgents/x_res)):
        temp_df = orderedYdata.iloc[ind:ind+x_res]
        temp_df.sort_values(["block_x"], axis=0, ascending=[True], inplace=True)
        temp_df.to_csv("csvOrderedFinal/" + "orderedFinal" + os.path.basename(fname), mode='a', index=False, header=False)
        ind += x_res

    # Add header to ordered csv
    header_list = ["curStep","t","block_x","block_y","p_cfd","u_cfd","v_cfd","p_pred","u_pred","v_pred"]
    orderedData = pd.read_csv("csvOrderedFinal/" + "orderedFinal" + os.path.basename(fname), header=None)
    orderedData.to_csv("csvOrderedFinal/" + "orderedFinal" + os.path.basename(fname), header=header_list, index=False)
    finalData = pd.read_csv("csvOrderedFinal/" + "orderedFinal" + os.path.basename(fname))

    p_cfd, u_cfd, v_cfd = np.zeros((y_res, x_res)), np.zeros((y_res, x_res)), np.zeros((y_res, x_res))
    p_pred, u_pred, v_pred = np.zeros((y_res, x_res)), np.zeros((y_res, x_res)), np.zeros((y_res, x_res))

    # Assign values to numpy data structure
    k = 0
    for j in range(y_res):
        for i in range(x_res):
            p_cfd[j, i] = finalData.iloc[k, 4]
            u_cfd[j, i] = finalData.iloc[k, 5]
            v_cfd[j, i] = finalData.iloc[k, 6]
            p_pred[j, i] = finalData.iloc[k, 7]
            u_pred[j, i] = finalData.iloc[k, 8]
            v_pred[j, i] = finalData.iloc[k, 9]
            k += 1
    
    return p_cfd, u_cfd, v_cfd, p_pred, u_pred, v_pred, fileID

raw_list = os.listdir(PATH)
filtered_list = list(filter(lambda f: f.endswith('.csv'), raw_list))

for fname in sorted(filtered_list):

    p_cfd, u_cfd, v_cfd, p_pred, u_pred, v_pred, fileID = orderAndAssign(PATH+fname)

    p_cfd_tot.append(p_cfd)
    u_cfd_tot.append(u_cfd)
    v_cfd_tot.append(v_cfd)
    p_pred_tot.append(p_pred)
    u_pred_tot.append(u_pred)
    v_pred_tot.append(v_pred)

    fileID_tot.append(fileID)

p_cfd_tot, p_pred_tot = np.array(p_cfd_tot, dtype=np.float32), np.array(p_pred_tot, dtype=np.float32)
u_cfd_tot, u_pred_tot = np.array(u_cfd_tot, dtype=np.float32), np.array(u_pred_tot, dtype=np.float32)
v_cfd_tot, v_pred_tot = np.array(v_cfd_tot, dtype=np.float32), np.array(v_pred_tot, dtype=np.float32)

quantity_list_cfd  = [p_cfd_tot, u_cfd_tot, v_cfd_tot]
quantity_list_pred = [p_pred_tot, u_pred_tot, v_pred_tot]

# Plotting and saving figures
for q_ind, (qty_cfd, qty_pred) in enumerate(zip(quantity_list_cfd, quantity_list_pred)):

    # Prepare figures
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
    fig.tight_layout(pad=2)
    timeHorizon = np.shape(p_cfd_tot)[0]

    # Finding global extrema over time horizon
    minValCFD,  maxValCFD  = [1e10, 1e10, 1e10], [0, 0, 0]
    minValPred, maxValPred = [1e10, 1e10, 1e10], [0, 0, 0]
    minValErr,  maxValErr  = [1e10, 1e10, 1e10], [0, 0, 0]
    
    for t in range(timeHorizon):

        minValCFD[q_ind] = min(np.amin(qty_cfd[t, :, :]), minValCFD[q_ind])
        maxValCFD[q_ind] = max(np.amax(qty_cfd[t, :, :]), maxValCFD[q_ind])

        minValPred[q_ind] = min(np.amin(qty_pred[t, :, :]), minValPred[q_ind])
        maxValPred[q_ind] = max(np.amax(qty_pred[t, :, :]), maxValPred[q_ind])

        minValErr[q_ind] = min(np.amin(abs(qty_cfd[t, :, :] - qty_pred[t, :, :])), minValErr[q_ind])
        maxValErr[q_ind] = max(np.amax(abs(qty_cfd[t, :, :] - qty_pred[t, :, :])), maxValErr[q_ind])

    minValBoth, maxValBoth = [0, 0, 0], [0, 0, 0]
    minValBoth[q_ind] = min(minValCFD[q_ind], minValPred[q_ind])
    maxValBoth[q_ind] = max(maxValCFD[q_ind], maxValPred[q_ind])

    # Plotting CFD, Prediction,Error for each physical quantity
    for t in range(timeHorizon):

        imageCFD = ax1.imshow(qty_cfd[t], cmap='seismic', interpolation='bilinear',\
                                norm=MidpointNormalize(midpoint=0, vmin=minValBoth[q_ind], vmax=maxValBoth[q_ind]))
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cbar1 = fig.colorbar(imageCFD, ax=ax1, cax=cax1)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_title('CFD')

        imagePred  = ax2.imshow(qty_pred[t], cmap='seismic', interpolation='bilinear',\
                                norm=MidpointNormalize(midpoint=0, vmin=minValBoth[q_ind], vmax=maxValBoth[q_ind]))
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cbar2 = fig.colorbar(imagePred, ax=ax2, cax=cax2)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_title('Prediction')

        imageErr = ax3.imshow(abs(qty_cfd[t] - qty_pred[t]), cmap='Reds', interpolation='bilinear',\
                            vmin=minValErr[q_ind], vmax=maxValErr[q_ind])
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        cbar3 = fig.colorbar(imageErr, ax=ax3, cax=cax3)
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_title("MAE")

        if q_ind==0: fig.savefig("p/" + fileID_tot[t] + ".jpg")
        elif q_ind==1: fig.savefig("u/" + fileID_tot[t] + ".jpg")
        elif q_ind==2: fig.savefig("v/" + fileID_tot[t] + ".jpg")
        else:
            print('sth dont work right')

        cbar1.remove()
        cbar2.remove()
        cbar3.remove()
