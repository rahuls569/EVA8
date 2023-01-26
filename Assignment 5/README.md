Question: 

![image](https://user-images.githubusercontent.com/79099957/214836764-4c5b9074-abbd-41e9-ac33-2bb66dde5e4f.png)

Response: 

A single model.py file that includes GN/LN/BN and takes an argument to decide which normalization to include

import torch.nn.functional as F

norm_type='GN'

class Net(nn.Module):

    def __init__(self, norm_type='GN', dropout_value = 0.1, num_groups =2):
    
        super(Net, self).__init__()
        self.norm_type= norm_type
        
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        if self.norm_type == 'GN':
            self.convblock2.add_module('norm', nn.GroupNorm(num_groups, num_channels=16))
        elif self.norm_type == 'LN':
            self.convblock2.add_module('norm', nn.LayerNorm([16, 24, 24]))
        elif self.norm_type == 'BN':
            self.convblock2.add_module('norm', nn.BatchNorm2d(16))

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
           
            nn.Dropout(dropout_value)
        ) # output_size = 10

        if self.norm_type == 'GN':
            self.convblock4.add_module('norm', nn.GroupNorm(num_groups, num_channels=16))
        elif self.norm_type == 'LN':
            self.convblock4.add_module('norm', nn.LayerNorm([16, 10, 10]))
        elif self.norm_type == 'BN':
            self.convblock4.add_module('norm', nn.BatchNorm2d(16))       




        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            
            nn.Dropout(dropout_value)
        ) # output_size = 8

        if self.norm_type == 'GN':
            self.convblock5.add_module('norm', nn.GroupNorm(num_groups, num_channels=8))
        elif self.norm_type == 'LN':
            self.convblock5.add_module('norm', nn.LayerNorm([8, 8, 8]))
        elif self.norm_type == 'BN':
            self.convblock5.add_module('norm', nn.BatchNorm2d(8))      

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
           
            nn.Dropout(dropout_value)
        ) # output_size = 6

        if self.norm_type == 'GN':
            self.convblock6.add_module('norm', nn.GroupNorm(num_groups, num_channels=16))
        elif self.norm_type == 'LN':
            self.convblock6.add_module('norm', nn.LayerNorm([16, 6, 6]))
        elif self.norm_type == 'BN':
            self.convblock6.add_module('norm', nn.BatchNorm2d(16))      


        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
           
            nn.Dropout(dropout_value)
        ) # output_size = 4

        if self.norm_type == 'GN':
            self.convblock7.add_module('norm', nn.GroupNorm(num_groups, num_channels=16))
        elif self.norm_type == 'LN':
            self.convblock7.add_module('norm', nn.LayerNorm([16, 4, 4]))
        elif self.norm_type == 'BN':
            self.convblock7.add_module('norm', nn.BatchNorm2d(16))  
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) 

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)

        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1,10)

        return F.log_softmax(x, dim=-1)

Training and testing:  

![image](https://user-images.githubusercontent.com/79099957/214838715-2042a7f0-a391-4784-b5e0-b8a723a792c3.png)

![image](https://user-images.githubusercontent.com/79099957/214838821-5fa5154e-d512-43a0-9552-a62151f32656.png)  
    
![image](https://user-images.githubusercontent.com/79099957/214838558-37a38004-9804-477c-929d-3a1cd1bfbb52.png)


First we select BN+ L1

Number of paramerter:

![image](https://user-images.githubusercontent.com/79099957/214839143-f6d753fc-037f-4977-862b-2ed6c7211532.png)

Logs: 

EPOCH: 0
Loss=0.39606931805610657 Batch_id=468 Accuracy=89.08: 100%|██████████| 469/469 [00:24<00:00, 19.36it/s]

Test set: Average loss: 0.0950, Accuracy: 9738/10000 (97.38%)

EPOCH: 1
Loss=0.3180012106895447 Batch_id=468 Accuracy=96.76: 100%|██████████| 469/469 [00:21<00:00, 21.36it/s]

Test set: Average loss: 0.0824, Accuracy: 9747/10000 (97.47%)

EPOCH: 2
Loss=0.32364338636398315 Batch_id=468 Accuracy=97.09: 100%|██████████| 469/469 [00:19<00:00, 24.48it/s]

Test set: Average loss: 0.0538, Accuracy: 9829/10000 (98.29%)

EPOCH: 3
Loss=0.27109137177467346 Batch_id=468 Accuracy=97.16: 100%|██████████| 469/469 [00:22<00:00, 20.84it/s]

Test set: Average loss: 0.0885, Accuracy: 9711/10000 (97.11%)

EPOCH: 4
Loss=0.2798900604248047 Batch_id=468 Accuracy=97.28: 100%|██████████| 469/469 [00:19<00:00, 24.36it/s]

Test set: Average loss: 0.0425, Accuracy: 9875/10000 (98.75%)

EPOCH: 5
Loss=0.281894713640213 Batch_id=468 Accuracy=97.33: 100%|██████████| 469/469 [00:19<00:00, 24.32it/s]

Test set: Average loss: 0.0757, Accuracy: 9770/10000 (97.70%)

EPOCH: 6
Loss=0.17723558843135834 Batch_id=468 Accuracy=98.37: 100%|██████████| 469/469 [00:19<00:00, 24.67it/s]

Test set: Average loss: 0.0244, Accuracy: 9934/10000 (99.34%)

EPOCH: 7
Loss=0.16349229216575623 Batch_id=468 Accuracy=98.51: 100%|██████████| 469/469 [00:19<00:00, 23.54it/s]

Test set: Average loss: 0.0250, Accuracy: 9919/10000 (99.19%)

EPOCH: 8
Loss=0.17407618463039398 Batch_id=468 Accuracy=98.57: 100%|██████████| 469/469 [00:19<00:00, 24.21it/s]

Test set: Average loss: 0.0248, Accuracy: 9926/10000 (99.26%)

EPOCH: 9
Loss=0.1956568956375122 Batch_id=468 Accuracy=98.51: 100%|██████████| 469/469 [00:19<00:00, 24.47it/s]

Test set: Average loss: 0.0247, Accuracy: 9924/10000 (99.24%)

EPOCH: 10
Loss=0.19010157883167267 Batch_id=468 Accuracy=98.38: 100%|██████████| 469/469 [00:19<00:00, 24.68it/s]

Test set: Average loss: 0.0233, Accuracy: 9934/10000 (99.34%)

EPOCH: 11
Loss=0.17108646035194397 Batch_id=468 Accuracy=98.53: 100%|██████████| 469/469 [00:19<00:00, 24.36it/s]

Test set: Average loss: 0.0228, Accuracy: 9933/10000 (99.33%)

EPOCH: 12
Loss=0.15552030503749847 Batch_id=468 Accuracy=98.67: 100%|██████████| 469/469 [00:19<00:00, 24.28it/s]

Test set: Average loss: 0.0207, Accuracy: 9939/10000 (99.39%)

EPOCH: 13
Loss=0.151715487241745 Batch_id=468 Accuracy=98.72: 100%|██████████| 469/469 [00:19<00:00, 24.54it/s]

Test set: Average loss: 0.0204, Accuracy: 9942/10000 (99.42%)

EPOCH: 14
Loss=0.13349154591560364 Batch_id=468 Accuracy=98.64: 100%|██████████| 469/469 [00:19<00:00, 24.33it/s]

Test set: Average loss: 0.0202, Accuracy: 9937/10000 (99.37%)

EPOCH: 15
Loss=0.1285482794046402 Batch_id=468 Accuracy=98.77: 100%|██████████| 469/469 [00:20<00:00, 22.98it/s]

Test set: Average loss: 0.0204, Accuracy: 9939/10000 (99.39%)

EPOCH: 16
Loss=0.19610695540905 Batch_id=468 Accuracy=98.71: 100%|██████████| 469/469 [00:19<00:00, 24.47it/s]

Test set: Average loss: 0.0197, Accuracy: 9939/10000 (99.39%)

EPOCH: 17
Loss=0.14778466522693634 Batch_id=468 Accuracy=98.74: 100%|██████████| 469/469 [00:18<00:00, 24.69it/s]

Test set: Average loss: 0.0207, Accuracy: 9939/10000 (99.39%)

EPOCH: 18
Loss=0.13853460550308228 Batch_id=468 Accuracy=98.79: 100%|██████████| 469/469 [00:19<00:00, 24.00it/s]

Test set: Average loss: 0.0202, Accuracy: 9943/10000 (99.43%)

EPOCH: 19
Loss=0.12828311324119568 Batch_id=468 Accuracy=98.75: 100%|██████████| 469/469 [00:19<00:00, 24.45it/s]

Test set: Average loss: 0.0204, Accuracy: 9939/10000 (99.39%)


Now select GN

Parameters: 

![image](https://user-images.githubusercontent.com/79099957/214840337-545043a5-2437-42f5-a578-4c5d6ab91cdc.png)

Logs:
EPOCH: 0
Loss=0.11952470988035202 Batch_id=468 Accuracy=88.70: 100%|██████████| 469/469 [00:22<00:00, 20.73it/s]

Test set: Average loss: 0.0725, Accuracy: 9790/10000 (97.90%)

EPOCH: 1
Loss=0.05127333104610443 Batch_id=468 Accuracy=97.30: 100%|██████████| 469/469 [00:17<00:00, 26.80it/s]

Test set: Average loss: 0.0425, Accuracy: 9876/10000 (98.76%)

EPOCH: 2
Loss=0.09954768419265747 Batch_id=468 Accuracy=97.97: 100%|██████████| 469/469 [00:17<00:00, 26.39it/s]

Test set: Average loss: 0.0352, Accuracy: 9877/10000 (98.77%)

EPOCH: 3
Loss=0.015369021333754063 Batch_id=468 Accuracy=98.11: 100%|██████████| 469/469 [00:17<00:00, 26.65it/s]

Test set: Average loss: 0.0317, Accuracy: 9898/10000 (98.98%)

EPOCH: 4
Loss=0.08783462643623352 Batch_id=468 Accuracy=98.29: 100%|██████████| 469/469 [00:17<00:00, 26.37it/s]

Test set: Average loss: 0.0292, Accuracy: 9907/10000 (99.07%)

EPOCH: 5
Loss=0.0622020848095417 Batch_id=468 Accuracy=98.42: 100%|██████████| 469/469 [00:18<00:00, 25.80it/s]

Test set: Average loss: 0.0316, Accuracy: 9906/10000 (99.06%)

EPOCH: 6
Loss=0.12990854680538177 Batch_id=468 Accuracy=98.81: 100%|██████████| 469/469 [00:17<00:00, 26.68it/s]

Test set: Average loss: 0.0237, Accuracy: 9928/10000 (99.28%)

EPOCH: 7
Loss=0.04067428782582283 Batch_id=468 Accuracy=98.81: 100%|██████████| 469/469 [00:17<00:00, 26.77it/s]

Test set: Average loss: 0.0236, Accuracy: 9929/10000 (99.29%)

EPOCH: 8
Loss=0.035340145230293274 Batch_id=468 Accuracy=98.84: 100%|██████████| 469/469 [00:18<00:00, 24.93it/s]

Test set: Average loss: 0.0231, Accuracy: 9928/10000 (99.28%)

EPOCH: 9
Loss=0.022839156910777092 Batch_id=468 Accuracy=98.86: 100%|██████████| 469/469 [00:17<00:00, 26.71it/s]

Test set: Average loss: 0.0236, Accuracy: 9932/10000 (99.32%)

EPOCH: 10
Loss=0.01934804953634739 Batch_id=468 Accuracy=98.88: 100%|██████████| 469/469 [00:17<00:00, 26.36it/s]

Test set: Average loss: 0.0223, Accuracy: 9932/10000 (99.32%)

EPOCH: 11
Loss=0.064816415309906 Batch_id=468 Accuracy=98.95: 100%|██████████| 469/469 [00:17<00:00, 26.40it/s]

Test set: Average loss: 0.0228, Accuracy: 9932/10000 (99.32%)

EPOCH: 12
Loss=0.02189166285097599 Batch_id=468 Accuracy=98.92: 100%|██████████| 469/469 [00:17<00:00, 26.22it/s]

Test set: Average loss: 0.0225, Accuracy: 9933/10000 (99.33%)

EPOCH: 13
Loss=0.014220948331058025 Batch_id=468 Accuracy=98.98: 100%|██████████| 469/469 [00:17<00:00, 26.55it/s]

Test set: Average loss: 0.0223, Accuracy: 9933/10000 (99.33%)

EPOCH: 14
Loss=0.016384756192564964 Batch_id=468 Accuracy=98.97: 100%|██████████| 469/469 [00:17<00:00, 26.75it/s]

Test set: Average loss: 0.0222, Accuracy: 9932/10000 (99.32%)

EPOCH: 15
Loss=0.006598433014005423 Batch_id=468 Accuracy=98.91: 100%|██████████| 469/469 [00:17<00:00, 27.02it/s]

Test set: Average loss: 0.0222, Accuracy: 9931/10000 (99.31%)

EPOCH: 16
Loss=0.014551709406077862 Batch_id=468 Accuracy=98.92: 100%|██████████| 469/469 [00:17<00:00, 26.43it/s]

Test set: Average loss: 0.0225, Accuracy: 9931/10000 (99.31%)

EPOCH: 17
Loss=0.029848357662558556 Batch_id=468 Accuracy=98.95: 100%|██████████| 469/469 [00:17<00:00, 27.06it/s]

Test set: Average loss: 0.0221, Accuracy: 9934/10000 (99.34%)

EPOCH: 18
Loss=0.010237353853881359 Batch_id=468 Accuracy=99.01: 100%|██████████| 469/469 [00:17<00:00, 27.26it/s]

Test set: Average loss: 0.0221, Accuracy: 9934/10000 (99.34%)

EPOCH: 19
Loss=0.010485522449016571 Batch_id=468 Accuracy=98.98: 100%|██████████| 469/469 [00:18<00:00, 25.33it/s]

Test set: Average loss: 0.0221, Accuracy: 9934/10000 (99.34%)

Now select LN

Parameters:

![image](https://user-images.githubusercontent.com/79099957/214842011-4f0d0fab-dd44-477c-84bc-d2670a7b2f45.png)

logs:

EPOCH: 0
Loss=0.09407278150320053 Batch_id=468 Accuracy=88.27: 100%|██████████| 469/469 [00:16<00:00, 27.90it/s]

Test set: Average loss: 0.0717, Accuracy: 9784/10000 (97.84%)

EPOCH: 1
Loss=0.11647193878889084 Batch_id=468 Accuracy=97.19: 100%|██████████| 469/469 [00:16<00:00, 27.75it/s]

Test set: Average loss: 0.0446, Accuracy: 9865/10000 (98.65%)

EPOCH: 2
Loss=0.05747978389263153 Batch_id=468 Accuracy=97.83: 100%|██████████| 469/469 [00:17<00:00, 26.63it/s]

Test set: Average loss: 0.0415, Accuracy: 9878/10000 (98.78%)

EPOCH: 3
Loss=0.07184985280036926 Batch_id=468 Accuracy=98.03: 100%|██████████| 469/469 [00:17<00:00, 27.27it/s]

Test set: Average loss: 0.0290, Accuracy: 9906/10000 (99.06%)

EPOCH: 4
Loss=0.04195895418524742 Batch_id=468 Accuracy=98.25: 100%|██████████| 469/469 [00:17<00:00, 27.17it/s]

Test set: Average loss: 0.0318, Accuracy: 9908/10000 (99.08%)

EPOCH: 5
Loss=0.022958099842071533 Batch_id=468 Accuracy=98.44: 100%|██████████| 469/469 [00:17<00:00, 26.33it/s]

Test set: Average loss: 0.0275, Accuracy: 9920/10000 (99.20%)

EPOCH: 6
Loss=0.050637681037187576 Batch_id=468 Accuracy=98.76: 100%|██████████| 469/469 [00:17<00:00, 26.74it/s]

Test set: Average loss: 0.0241, Accuracy: 9927/10000 (99.27%)

EPOCH: 7
Loss=0.06796734780073166 Batch_id=468 Accuracy=98.81: 100%|██████████| 469/469 [00:17<00:00, 26.97it/s]

Test set: Average loss: 0.0244, Accuracy: 9921/10000 (99.21%)

EPOCH: 8
Loss=0.09877041727304459 Batch_id=468 Accuracy=98.87: 100%|██████████| 469/469 [00:17<00:00, 27.02it/s]

Test set: Average loss: 0.0234, Accuracy: 9930/10000 (99.30%)

EPOCH: 9
Loss=0.006630800198763609 Batch_id=468 Accuracy=98.84: 100%|██████████| 469/469 [00:17<00:00, 27.07it/s]

Test set: Average loss: 0.0229, Accuracy: 9926/10000 (99.26%)

EPOCH: 10
Loss=0.014936997555196285 Batch_id=468 Accuracy=98.86: 100%|██████████| 469/469 [00:17<00:00, 27.22it/s]

Test set: Average loss: 0.0233, Accuracy: 9928/10000 (99.28%)

EPOCH: 11
Loss=0.04758158326148987 Batch_id=468 Accuracy=98.84: 100%|██████████| 469/469 [00:17<00:00, 27.06it/s]

Test set: Average loss: 0.0222, Accuracy: 9932/10000 (99.32%)

EPOCH: 12
Loss=0.02268517203629017 Batch_id=468 Accuracy=98.92: 100%|██████████| 469/469 [00:17<00:00, 26.93it/s]

Test set: Average loss: 0.0224, Accuracy: 9928/10000 (99.28%)

EPOCH: 13
Loss=0.02082117088139057 Batch_id=468 Accuracy=98.93: 100%|██████████| 469/469 [00:17<00:00, 27.19it/s]

Test set: Average loss: 0.0224, Accuracy: 9927/10000 (99.27%)

EPOCH: 14
Loss=0.009491879492998123 Batch_id=468 Accuracy=98.93: 100%|██████████| 469/469 [00:17<00:00, 27.10it/s]

Test set: Average loss: 0.0224, Accuracy: 9926/10000 (99.26%)

EPOCH: 15
Loss=0.11348015815019608 Batch_id=468 Accuracy=98.87: 100%|██████████| 469/469 [00:17<00:00, 26.99it/s]

Test set: Average loss: 0.0224, Accuracy: 9926/10000 (99.26%)

EPOCH: 16
Loss=0.04696537181735039 Batch_id=468 Accuracy=98.94: 100%|██████████| 469/469 [00:18<00:00, 25.33it/s]

Test set: Average loss: 0.0225, Accuracy: 9929/10000 (99.29%)

EPOCH: 17
Loss=0.06928838789463043 Batch_id=468 Accuracy=98.91: 100%|██████████| 469/469 [00:17<00:00, 27.28it/s]

Test set: Average loss: 0.0224, Accuracy: 9927/10000 (99.27%)

EPOCH: 18
Loss=0.01385620329529047 Batch_id=468 Accuracy=98.94: 100%|██████████| 469/469 [00:17<00:00, 27.24it/s]

Test set: Average loss: 0.0224, Accuracy: 9928/10000 (99.28%)

EPOCH: 19
Loss=0.07492489367723465 Batch_id=468 Accuracy=98.96: 100%|██████████| 469/469 [00:17<00:00, 27.47it/s]

Test set: Average loss: 0.0224, Accuracy: 9928/10000 (99.28%)


how to perform the 3 normalizations techniques that we covered(cannot use values from the excel sheet shared)

![image](https://user-images.githubusercontent.com/79099957/214885213-e19a2181-f53a-4d3c-993e-f9b245b7a5ad.png)

In Batch normalization, we are taking batches in each images and performing mean and variance
In layer normalization, we are taking layer of each images and performing mean and variance
in group normalization, we are taking two group of each image and performing mean and variance

add all your graphs

![image](https://user-images.githubusercontent.com/79099957/214843182-ee939443-d4e7-40b3-85b7-c122929d6b1d.png)


your 3 collection-of-misclassified-images 
Using LN

![image](https://user-images.githubusercontent.com/79099957/214853344-ace8d31b-db0d-4323-b910-979d80ce9173.png)

USing GN

![image](https://user-images.githubusercontent.com/79099957/214856173-d075ce7d-78a7-4f2d-952b-276f0d825395.png)

Using BN+L1

![image](https://user-images.githubusercontent.com/79099957/214858858-210ffae8-d81c-4384-a3eb-7e7d46effa39.png)






