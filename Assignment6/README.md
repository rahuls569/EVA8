Assignment : 6 
Run this network Links to an external site..

Fix the network above:

change the code such that it uses GPU and change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)

total RF must be more than 44

one of the layers must use Depthwise Separable Convolution

one of the layers must use Dilated Convolution

use GAP (compulsory):- add FC after GAP to target #of classes (optional)

use albumentation library and apply:

horizontal flip

shiftScaleRotate

coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

upload to Github

Attempt S6-Assignment Solution.

Questions in the Assignment QnA are:

copy paste your model code from your model.py file (full code) [125]

copy paste output of torchsummary [125]

copy-paste the code where you implemented albumentation transformation for all three transformations [125]

copy paste your training log (you must be running validation/text after each Epoch [125]

Share the link for your README.md file. [200]



Response:


Model

![image](https://user-images.githubusercontent.com/79099957/217046537-778abac2-4aff-45df-a5dd-a4c6988d15bf.png)


 Torch summary
 
 ![image](https://user-images.githubusercontent.com/79099957/217046683-11de6de2-ffc6-4230-8bfb-dce3d91693a9.png)


Copy-paste the code where you implemented albumentation transformation for all the three transformation as mentioned in the assignment:

![image](https://user-images.githubusercontent.com/79099957/217047765-93f8df77-2c19-4e59-96a7-ed08920be496.png)

![image](https://user-images.githubusercontent.com/79099957/217047897-babe1ccb-39cd-4b16-afd3-a277d22b8ab4.png)





Copy-paste your training log (you must be running validation/test check after each Epoch:

EPOCH: 0
Loss=2.1536524295806885 Batch_id=390 Accuracy=12.22: 100%|██████████| 391/391 [00:13<00:00, 29.35it/s]

Test set: Average loss: 0.0167, Accuracy: 2007/10000 (20.07%)

EPOCH: 1
Loss=1.6432708501815796 Batch_id=390 Accuracy=25.50: 100%|██████████| 391/391 [00:20<00:00, 18.62it/s]

Test set: Average loss: 0.0147, Accuracy: 3004/10000 (30.04%)

EPOCH: 2
Loss=1.6514465808868408 Batch_id=390 Accuracy=33.81: 100%|██████████| 391/391 [00:12<00:00, 30.49it/s]

Test set: Average loss: 0.0131, Accuracy: 3891/10000 (38.91%)

EPOCH: 3
Loss=1.5912857055664062 Batch_id=390 Accuracy=38.73: 100%|██████████| 391/391 [00:12<00:00, 30.76it/s]

Test set: Average loss: 0.0131, Accuracy: 3981/10000 (39.81%)

EPOCH: 4
Loss=1.5982099771499634 Batch_id=390 Accuracy=42.03: 100%|██████████| 391/391 [00:12<00:00, 30.37it/s]

Test set: Average loss: 0.0119, Accuracy: 4363/10000 (43.63%)

EPOCH: 5
Loss=1.4570631980895996 Batch_id=390 Accuracy=44.60: 100%|██████████| 391/391 [00:12<00:00, 30.09it/s]

Test set: Average loss: 0.0116, Accuracy: 4573/10000 (45.73%)

EPOCH: 6
Loss=1.5265867710113525 Batch_id=390 Accuracy=48.28: 100%|██████████| 391/391 [00:12<00:00, 30.15it/s]

Test set: Average loss: 0.0111, Accuracy: 4847/10000 (48.47%)

EPOCH: 7
Loss=1.3426824808120728 Batch_id=390 Accuracy=48.70: 100%|██████████| 391/391 [00:13<00:00, 28.40it/s]

Test set: Average loss: 0.0110, Accuracy: 4884/10000 (48.84%)

EPOCH: 8
Loss=1.3750479221343994 Batch_id=390 Accuracy=48.99: 100%|██████████| 391/391 [00:12<00:00, 30.16it/s]

Test set: Average loss: 0.0110, Accuracy: 4935/10000 (49.35%)

EPOCH: 9
Loss=1.2958134412765503 Batch_id=390 Accuracy=49.51: 100%|██████████| 391/391 [00:13<00:00, 29.40it/s]

Test set: Average loss: 0.0110, Accuracy: 4950/10000 (49.50%)

EPOCH: 10
Loss=1.2513818740844727 Batch_id=390 Accuracy=49.68: 100%|██████████| 391/391 [00:13<00:00, 28.34it/s]

Test set: Average loss: 0.0109, Accuracy: 4945/10000 (49.45%)

EPOCH: 11
Loss=1.431261658668518 Batch_id=390 Accuracy=49.90: 100%|██████████| 391/391 [00:13<00:00, 28.37it/s]

Test set: Average loss: 0.0109, Accuracy: 4967/10000 (49.67%)

EPOCH: 12
Loss=1.4345617294311523 Batch_id=390 Accuracy=50.39: 100%|██████████| 391/391 [00:13<00:00, 28.56it/s]

Test set: Average loss: 0.0108, Accuracy: 4990/10000 (49.90%)

EPOCH: 13
Loss=1.4503543376922607 Batch_id=390 Accuracy=50.56: 100%|██████████| 391/391 [00:13<00:00, 28.69it/s]

Test set: Average loss: 0.0108, Accuracy: 5012/10000 (50.12%)

EPOCH: 14
Loss=1.23600172996521 Batch_id=390 Accuracy=50.57: 100%|██████████| 391/391 [00:13<00:00, 28.60it/s]

Test set: Average loss: 0.0107, Accuracy: 5011/10000 (50.11%)

EPOCH: 15
Loss=1.4274652004241943 Batch_id=390 Accuracy=50.64: 100%|██████████| 391/391 [00:14<00:00, 27.66it/s]

Test set: Average loss: 0.0108, Accuracy: 5019/10000 (50.19%)

EPOCH: 16
Loss=1.3835796117782593 Batch_id=390 Accuracy=50.70: 100%|██████████| 391/391 [00:14<00:00, 27.68it/s]

Test set: Average loss: 0.0108, Accuracy: 5016/10000 (50.16%)

EPOCH: 17
Loss=1.4479032754898071 Batch_id=390 Accuracy=50.56: 100%|██████████| 391/391 [00:13<00:00, 28.16it/s]

Test set: Average loss: 0.0107, Accuracy: 5027/10000 (50.27%)

EPOCH: 18
Loss=1.3205052614212036 Batch_id=390 Accuracy=50.82: 100%|██████████| 391/391 [00:13<00:00, 28.50it/s]

Test set: Average loss: 0.0108, Accuracy: 5024/10000 (50.24%)

EPOCH: 19
Loss=1.3291043043136597 Batch_id=390 Accuracy=50.80: 100%|██████████| 391/391 [00:13<00:00, 28.89it/s]

Test set: Average loss: 0.0108, Accuracy: 5028/10000 (50.28%)

EPOCH: 20
Loss=1.3442044258117676 Batch_id=390 Accuracy=50.79: 100%|██████████| 391/391 [00:14<00:00, 27.82it/s]

Test set: Average loss: 0.0108, Accuracy: 5023/10000 (50.23%)

EPOCH: 21
Loss=1.361838936805725 Batch_id=390 Accuracy=50.79: 100%|██████████| 391/391 [00:14<00:00, 27.54it/s]

Test set: Average loss: 0.0107, Accuracy: 5026/10000 (50.26%)

EPOCH: 22
Loss=1.2654082775115967 Batch_id=390 Accuracy=50.80: 100%|██████████| 391/391 [00:14<00:00, 26.88it/s]

Test set: Average loss: 0.0108, Accuracy: 5023/10000 (50.23%)

EPOCH: 23
Loss=1.4321197271347046 Batch_id=390 Accuracy=50.75: 100%|██████████| 391/391 [00:14<00:00, 26.67it/s]

Test set: Average loss: 0.0108, Accuracy: 5017/10000 (50.17%)

EPOCH: 24
Loss=1.2190752029418945 Batch_id=390 Accuracy=50.77: 100%|██████████| 391/391 [00:13<00:00, 28.42it/s]

Test set: Average loss: 0.0107, Accuracy: 5020/10000 (50.20%)




















