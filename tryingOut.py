
import os
'''
Vis("/home/tzikos/Desktop/Data/Berts downsapled/pre/test/AVNRT pre-D659DA5A-12D1-4C15-A28A-6B3B7C6BA135_2F39RYB8MXRN_20090208_1.npy",
      "/home/tzikos/Desktop/Data/Berts downsapled/pre/test/normal pre-E6C4BEFB-02C5-4F61-B9F7-2D748DE7322E_QEPNJX5YU2FN_20120723_1.npy",
      "Berts-AVNRT-Normal-Downsampled",
      "Berts AVNRT - Normal Downsampled")

Vis("/home/tzikos/Desktop/Data/Berts no noise/pre/test/AVNRT pre-D659DA5A-12D1-4C15-A28A-6B3B7C6BA135_2F39RYB8MXRN_20090208_1.npy",
      "/home/tzikos/Desktop/Data/Berts no noise/pre/test/normal pre-E6C4BEFB-02C5-4F61-B9F7-2D748DE7322E_QEPNJX5YU2FN_20120723_1.npy",
      "Berts-AVNRT-Normal-Denoised",
      "Berts AVNRT - Normal Denoised")

Vis("/home/tzikos/Desktop/Data/Berts gaussian/pre/test/AVNRT pre-D659DA5A-12D1-4C15-A28A-6B3B7C6BA135_2F39RYB8MXRN_20090208_1.npy",
      "/home/tzikos/Desktop/Data/Berts gaussian/pre/test/normal pre-E6C4BEFB-02C5-4F61-B9F7-2D748DE7322E_QEPNJX5YU2FN_20120723_1.npy",
      "Berts-AVNRT-Normal-Normalised",
      "Berts AVNRT - Normal Normalised")

Vis("/home/tzikos/Desktop/Data/Berts torch/pre/test/AVNRT pre-D659DA5A-12D1-4C15-A28A-6B3B7C6BA135_2F39RYB8MXRN_20090208_1-1.npy",
      "/home/tzikos/Desktop/Data/Berts torch/pre/test/normal pre-E6C4BEFB-02C5-4F61-B9F7-2D748DE7322E_QEPNJX5YU2FN_20120723_1-1.npy",
      "Berts-AVNRT-Normal-Leads should be missing",
      "Berts AVNRT - Normal Leads should be missing")
'''


#checkBertMissing("/home/tzikos/Desktop/Data/Berts")
#findDuplicatePatients("/home/tzikos/Desktop/Data/Berts/pre")
#findDuplicatePatients("/home/tzikos/Desktop/Data/Berts/tachy")

#checkStats("/home/tzikos/Desktop/Data/Berts downsapled", "pre")
#EXPERIMENT = 'tachy'
#expList = [f'normal {EXPERIMENT}', f'AVNRT {EXPERIMENT}', f'AVRT {EXPERIMENT}', f'concealed {EXPERIMENT}', f'EAT {EXPERIMENT}']
#tableCreator(expList)
#EXPERIMENT = 'pre'
#expList = [f'normal {EXPERIMENT}', f'AVNRT {EXPERIMENT}', f'AVRT {EXPERIMENT}', f'concealed {EXPERIMENT}', f'EAT {EXPERIMENT}']
#tableCreator(expList)

#dataLeadStatsVisPerLead()


print(os.listdir("/home/tzikos/Desktop/Data/Berts final/tachy/"))