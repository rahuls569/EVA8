Assignment:

1. Write a custom ResNet architecture for CIFAR10 that has the following architecture:
    1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
    2. Layer1 -
      1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
      2. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
      3. Add(X, R1)
    3. Layer 2 -
      1. Conv 3x3 [256k]
      2. MaxPooling2D
      3. BN
      4. ReLU
    4. Layer 3 -
      1. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
      2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
      3. Add(X, R2)
    5. MaxPooling with Kernel Size 4
    6. FC Layer 
    7. SoftMax
2. Uses One Cycle Policy such that:
    1. Total Epochs = 24
    2. Max at Epoch = 5
    3. LRMIN = FIND
    4. LRMAX = FIND
    5. NO Annihilation
3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
4. Batch size = 512
5. Target Accuracy: 90% (93.8% quadruple scores). 
6. NO score if your code is not modular. Your collab must be importing your GitHub package, and then just running the model. I should be able to find the custom_resnet.py model in your GitHub repo that you'd be training. 
7. Once done, proceed to answer the Assignment-Solution page. 
 
 
Response:

Custom Resnet Model is given as:

![image](https://user-images.githubusercontent.com/79099957/221217142-268dfb66-15b4-4924-85c6-64c0c1bf9623.png)

![image](https://user-images.githubusercontent.com/79099957/221217242-3673cdf5-2761-47ca-8c86-ae7616e9a8fa.png)

![image](https://user-images.githubusercontent.com/79099957/221217375-9b855456-5933-4429-a21a-59155234de46.png)

Code used to train the model (where you defined epochs, criterion, LRMIN/MAX and other details):

![image](https://user-images.githubusercontent.com/79099957/221217973-270c19cb-3cd9-45f0-b83b-eb27c1fc04d8.png)

Training logs (must show test accuracy) and only 24 EPOCHS:
EPOCH: 1 (LR: 0.0028000000000000004)
Batch_id=97 Loss=2.10311 Accuracy=35.07%: 100%|██████████| 98/98 [00:28<00:00,  3.45it/s]

Test set: Average loss: 2.0278, Accuracy: 4303/10000 (43.03%)

EPOCH: 2 (LR: 0.016289111376931526)
Batch_id=97 Loss=1.97383 Accuracy=48.31%: 100%|██████████| 98/98 [00:26<00:00,  3.67it/s]

Test set: Average loss: 1.9714, Accuracy: 4862/10000 (48.62%)

EPOCH: 3 (LR: 0.029778222753863052)
Batch_id=97 Loss=1.87217 Accuracy=58.67%: 100%|██████████| 98/98 [00:27<00:00,  3.60it/s]

Test set: Average loss: 1.8297, Accuracy: 6300/10000 (63.00%)

EPOCH: 4 (LR: 0.043267334130794574)
Batch_id=97 Loss=1.81944 Accuracy=63.95%: 100%|██████████| 98/98 [00:26<00:00,  3.67it/s]

Test set: Average loss: 1.8182, Accuracy: 6410/10000 (64.10%)

EPOCH: 5 (LR: 0.0567564455077261)
Batch_id=97 Loss=1.78524 Accuracy=67.37%: 100%|██████████| 98/98 [00:26<00:00,  3.64it/s]

Test set: Average loss: 1.8066, Accuracy: 6517/10000 (65.17%)

EPOCH: 6 (LR: 0.06993298737373738)
Batch_id=97 Loss=1.74001 Accuracy=71.99%: 100%|██████████| 98/98 [00:27<00:00,  3.60it/s]

Test set: Average loss: 1.7352, Accuracy: 7252/10000 (72.52%)

EPOCH: 7 (LR: 0.06625180050505051)
Batch_id=97 Loss=1.70615 Accuracy=75.36%: 100%|██████████| 98/98 [00:27<00:00,  3.55it/s]

Test set: Average loss: 1.7034, Accuracy: 7576/10000 (75.76%)

EPOCH: 8 (LR: 0.06257061363636364)
Batch_id=97 Loss=1.68079 Accuracy=77.92%: 100%|██████████| 98/98 [00:27<00:00,  3.56it/s]

Test set: Average loss: 1.6990, Accuracy: 7619/10000 (76.19%)

EPOCH: 9 (LR: 0.05888942676767677)
Batch_id=97 Loss=1.67079 Accuracy=78.98%: 100%|██████████| 98/98 [00:27<00:00,  3.58it/s]

Test set: Average loss: 1.6791, Accuracy: 7819/10000 (78.19%)

EPOCH: 10 (LR: 0.0552082398989899)
Batch_id=97 Loss=1.65795 Accuracy=80.21%: 100%|██████████| 98/98 [00:27<00:00,  3.56it/s]

Test set: Average loss: 1.6704, Accuracy: 7892/10000 (78.92%)

EPOCH: 11 (LR: 0.051527053030303034)
Batch_id=97 Loss=1.64497 Accuracy=81.53%: 100%|██████████| 98/98 [00:27<00:00,  3.55it/s]

Test set: Average loss: 1.6411, Accuracy: 8210/10000 (82.10%)

EPOCH: 12 (LR: 0.047845866161616166)
Batch_id=97 Loss=1.63327 Accuracy=82.76%: 100%|██████████| 98/98 [00:27<00:00,  3.55it/s]

Test set: Average loss: 1.6401, Accuracy: 8206/10000 (82.06%)

EPOCH: 13 (LR: 0.0441646792929293)
Batch_id=97 Loss=1.62314 Accuracy=83.79%: 100%|██████████| 98/98 [00:27<00:00,  3.53it/s]

Test set: Average loss: 1.6346, Accuracy: 8261/10000 (82.61%)

EPOCH: 14 (LR: 0.04048349242424243)
Batch_id=97 Loss=1.61327 Accuracy=84.77%: 100%|██████████| 98/98 [00:27<00:00,  3.52it/s]

Test set: Average loss: 1.6389, Accuracy: 8230/10000 (82.30%)

EPOCH: 15 (LR: 0.03680230555555555)
Batch_id=97 Loss=1.60585 Accuracy=85.49%: 100%|██████████| 98/98 [00:27<00:00,  3.56it/s]

Test set: Average loss: 1.6235, Accuracy: 8377/10000 (83.77%)

EPOCH: 16 (LR: 0.033121118686868685)
Batch_id=97 Loss=1.59922 Accuracy=86.17%: 100%|██████████| 98/98 [00:27<00:00,  3.57it/s]

Test set: Average loss: 1.6153, Accuracy: 8456/10000 (84.56%)

EPOCH: 17 (LR: 0.029439931818181823)
Batch_id=97 Loss=1.59164 Accuracy=86.97%: 100%|██████████| 98/98 [00:27<00:00,  3.50it/s]

Test set: Average loss: 1.6075, Accuracy: 8546/10000 (85.46%)

EPOCH: 18 (LR: 0.025758744949494948)
Batch_id=97 Loss=1.58210 Accuracy=87.92%: 100%|██████████| 98/98 [00:27<00:00,  3.53it/s]

Test set: Average loss: 1.6040, Accuracy: 8572/10000 (85.72%)

EPOCH: 19 (LR: 0.022077558080808087)
Batch_id=97 Loss=1.57785 Accuracy=88.32%: 100%|██████████| 98/98 [00:27<00:00,  3.55it/s]

Test set: Average loss: 1.6023, Accuracy: 8596/10000 (85.96%)

EPOCH: 20 (LR: 0.01839637121212121)
Batch_id=97 Loss=1.57284 Accuracy=88.84%: 100%|██████████| 98/98 [00:27<00:00,  3.51it/s]

Test set: Average loss: 1.5960, Accuracy: 8659/10000 (86.59%)

EPOCH: 21 (LR: 0.01471518434343435)
Batch_id=97 Loss=1.56274 Accuracy=89.86%: 100%|██████████| 98/98 [00:27<00:00,  3.56it/s]

Test set: Average loss: 1.5869, Accuracy: 8741/10000 (87.41%)

EPOCH: 22 (LR: 0.011033997474747474)
Batch_id=97 Loss=1.55621 Accuracy=90.52%: 100%|██████████| 98/98 [00:27<00:00,  3.51it/s]

Test set: Average loss: 1.5816, Accuracy: 8808/10000 (88.08%)

EPOCH: 23 (LR: 0.007352810606060606)
Batch_id=97 Loss=1.55077 Accuracy=91.02%: 100%|██████████| 98/98 [00:27<00:00,  3.54it/s]

Test set: Average loss: 1.5795, Accuracy: 8825/10000 (88.25%)

EPOCH: 24 (LR: 0.0036716237373737304)
Batch_id=97 Loss=1.54562 Accuracy=91.63%: 100%|██████████| 98/98 [00:27<00:00,  3.55it/s]

Test set: Average loss: 1.5743, Accuracy: 8876/10000 (88.76%)

Code used for image augmentation of your images:

![image](https://user-images.githubusercontent.com/79099957/221219230-66872671-bd98-4177-8575-d013e5b9db54.png)

![image](https://user-images.githubusercontent.com/79099957/221219382-44fc50fc-5917-4a81-99a2-764ff9a35dd4.png)

Train_loader images:

![image](https://user-images.githubusercontent.com/79099957/221221058-47beb6d0-ff3d-459a-96cd-93bd807e31af.png)



GRAPH LOSS and ACCURACY:
![image](https://user-images.githubusercontent.com/79099957/221220919-6d59f269-3995-4e3d-a2a7-e73e943efe22.png)




















