KWS Adversarial Attack with CGAN
================================

This is the implemention of *Fast Speech Adversarial Example Generation for Keyword Spotting System with Conditional GAN* 

Setup
--------------------------
   The pre-trained victim model is available for use and can be downloaded from [here](https://drive.google.com/drive/folders/1b_CpcS6Yf3dpcDYKqIt1WSSr2y3B-gI9?usp=sharing)
    
   The pre-trained target(generator) model is available for use and can be downloaded from
   
   single model: link:https://pan.baidu.com/s/1xLJqCljv1mjDSiMZB8mrGw  fetech code：gk5c 
   
   multi-model ensemble: link：https://pan.baidu.com/s/1geJxrXoJxVYgLIyX5c3q0A  fetech code：8h7s 
   
   Notice: The directory structure of the checkpoints file is as follows: 

    |-- checkpoints
        |-- vgg19.pth	
        |-- resnet18.pth
        |-- resnext29_8_64.pth
        |-- dpn92.pth
        |-- densenet_bc_250_24.pth
        |-- wideResNet28_10_9414.pth

The folder structure of generated adversarial examples is as follows:

```
|-- output
    |-- wideresnet
    	|--target_yes
    		|--generated
    			|-- gen
	    			|-- no
    					|-- fake_yes2no_xxxx.wav
    			|-- real
    				|-- no
    					|-- real_yes2no_xxxx.wav
    	|--target_no
    		|--generated
    			|-- gen
    				|-- yes
    					|-- fake_no2yes_xxxx.wav
    			|-- real
				    |-- yes
    					|-- real_no2yes_xxxx.wav
```



Attack Evaluation
--------------------------

To run:

  python test_gan.py --data_dir original_speech.wav  --target yes --checkpoint checkpoints
