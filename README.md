# Vechile type and make detection ussing transfer learning and flask deployment.

Objective of this project was to train a nural network model ussing transfer learning and deploy it in aws creating a flask app.

# Installations
Python libraries:
1. Numpy
2. pandas
3. tensorflow
4. keras
5. flask

Data Set:
Can be downloaded from kagle or the official stanfor website:
kagle link: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset

The Cars dataset comprises a collection of 16,185 images, representing 196 distinct classes of cars. The dataset is divided into two subsets: a training set containing 8,144 images and a testing set containing 8,041 images. The split of images within each class is approximately evenly distributed, ensuring a balanced representation. The classes in the dataset typically correspond to the Make, Model, and Year of the cars, such as the 2012 Tesla Model S or the 2012 BMW M3 coupe.

# Future Work
Reduce the size of the model ussing various optimisation techniques and build a scallable web app extending the flask aws deployment. 
# Note:
As per the current stting the app runs in the port 8080 so if you deploying it in the aws you will have to use ssh tunning to access the portal.
# Model Architecture
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_2 (InputLayer)           [(None, 224, 224, 3  0           []                               
                                )]                                                                
                                                                                                  
 conv1_pad (ZeroPadding2D)      (None, 230, 230, 3)  0           ['input_2[0][0]']                
                                                                                                  
 conv1_conv (Conv2D)            (None, 112, 112, 64  9472        ['conv1_pad[0][0]']              
                                )                                                                 
                                                                                                  
 pool1_pad (ZeroPadding2D)      (None, 114, 114, 64  0           ['conv1_conv[0][0]']             
                                )                                                                 
                                                                                                  
 pool1_pool (MaxPooling2D)      (None, 56, 56, 64)   0           ['pool1_pad[0][0]']              
                                                                                                  
 conv2_block1_preact_bn (BatchN  (None, 56, 56, 64)  256         ['pool1_pool[0][0]']             
 ormalization)                                                                                    
                                                                                                  
 conv2_block1_preact_relu (Acti  (None, 56, 56, 64)  0           ['conv2_block1_preact_bn[0][0]'] 
 vation)                                                                                          
                                                                                                  
 conv2_block1_1_conv (Conv2D)   (None, 56, 56, 64)   4096        ['conv2_block1_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv2_block1_1_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block1_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block1_1_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block1_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block1_2_pad (ZeroPaddin  (None, 58, 58, 64)  0           ['conv2_block1_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv2_block1_2_conv (Conv2D)   (None, 56, 56, 64)   36864       ['conv2_block1_2_pad[0][0]']     
                                                                                                  
 conv2_block1_2_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block1_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block1_2_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block1_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block1_0_conv (Conv2D)   (None, 56, 56, 256)  16640       ['conv2_block1_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv2_block1_3_conv (Conv2D)   (None, 56, 56, 256)  16640       ['conv2_block1_2_relu[0][0]']    
                                                                                                  
 conv2_block1_out (Add)         (None, 56, 56, 256)  0           ['conv2_block1_0_conv[0][0]',    
                                                                  'conv2_block1_3_conv[0][0]']    
                                                                                                  
 conv2_block2_preact_bn (BatchN  (None, 56, 56, 256)  1024       ['conv2_block1_out[0][0]']       
 ormalization)                                                                                    
                                                                                                  
 conv2_block2_preact_relu (Acti  (None, 56, 56, 256)  0          ['conv2_block2_preact_bn[0][0]'] 
 vation)                                                                                          
                                                                                                  
 conv2_block2_1_conv (Conv2D)   (None, 56, 56, 64)   16384       ['conv2_block2_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv2_block2_1_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block2_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block2_1_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block2_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block2_2_pad (ZeroPaddin  (None, 58, 58, 64)  0           ['conv2_block2_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv2_block2_2_conv (Conv2D)   (None, 56, 56, 64)   36864       ['conv2_block2_2_pad[0][0]']     
                                                                                                  
 conv2_block2_2_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block2_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block2_2_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block2_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block2_3_conv (Conv2D)   (None, 56, 56, 256)  16640       ['conv2_block2_2_relu[0][0]']    
                                                                                                  
 conv2_block2_out (Add)         (None, 56, 56, 256)  0           ['conv2_block1_out[0][0]',       
                                                                  'conv2_block2_3_conv[0][0]']    
                                                                                                  
 conv2_block3_preact_bn (BatchN  (None, 56, 56, 256)  1024       ['conv2_block2_out[0][0]']       
 ormalization)                                                                                    
                                                                                                  
 conv2_block3_preact_relu (Acti  (None, 56, 56, 256)  0          ['conv2_block3_preact_bn[0][0]'] 
 vation)                                                                                          
                                                                                                  
 conv2_block3_1_conv (Conv2D)   (None, 56, 56, 64)   16384       ['conv2_block3_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv2_block3_1_bn (BatchNormal  (None, 56, 56, 64)  256         ['conv2_block3_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block3_1_relu (Activatio  (None, 56, 56, 64)  0           ['conv2_block3_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv2_block3_2_pad (ZeroPaddin  (None, 58, 58, 64)  0           ['conv2_block3_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv2_block3_2_conv (Conv2D)   (None, 28, 28, 64)   36864       ['conv2_block3_2_pad[0][0]']     
                                                                                                  
 conv2_block3_2_bn (BatchNormal  (None, 28, 28, 64)  256         ['conv2_block3_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv2_block3_2_relu (Activatio  (None, 28, 28, 64)  0           ['conv2_block3_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 max_pooling2d_3 (MaxPooling2D)  (None, 28, 28, 256)  0          ['conv2_block2_out[0][0]']       
                                                                                                  
 conv2_block3_3_conv (Conv2D)   (None, 28, 28, 256)  16640       ['conv2_block3_2_relu[0][0]']    
                                                                                                  
 conv2_block3_out (Add)         (None, 28, 28, 256)  0           ['max_pooling2d_3[0][0]',        
                                                                  'conv2_block3_3_conv[0][0]']    
                                                                                                  
 conv3_block1_preact_bn (BatchN  (None, 28, 28, 256)  1024       ['conv2_block3_out[0][0]']       
 ormalization)                                                                                    
                                                                                                  
 conv3_block1_preact_relu (Acti  (None, 28, 28, 256)  0          ['conv3_block1_preact_bn[0][0]'] 
 vation)                                                                                          
                                                                                                  
 conv3_block1_1_conv (Conv2D)   (None, 28, 28, 128)  32768       ['conv3_block1_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv3_block1_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block1_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block1_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block1_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block1_2_pad (ZeroPaddin  (None, 30, 30, 128)  0          ['conv3_block1_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv3_block1_2_conv (Conv2D)   (None, 28, 28, 128)  147456      ['conv3_block1_2_pad[0][0]']     
                                                                                                  
 conv3_block1_2_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block1_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block1_2_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block1_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block1_0_conv (Conv2D)   (None, 28, 28, 512)  131584      ['conv3_block1_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv3_block1_3_conv (Conv2D)   (None, 28, 28, 512)  66048       ['conv3_block1_2_relu[0][0]']    
                                                                                                  
 conv3_block1_out (Add)         (None, 28, 28, 512)  0           ['conv3_block1_0_conv[0][0]',    
                                                                  'conv3_block1_3_conv[0][0]']    
                                                                                                  
 conv3_block2_preact_bn (BatchN  (None, 28, 28, 512)  2048       ['conv3_block1_out[0][0]']       
 ormalization)                                                                                    
                                                                                                  
 conv3_block2_preact_relu (Acti  (None, 28, 28, 512)  0          ['conv3_block2_preact_bn[0][0]'] 
 vation)                                                                                          
                                                                                                  
 conv3_block2_1_conv (Conv2D)   (None, 28, 28, 128)  65536       ['conv3_block2_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv3_block2_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block2_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block2_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block2_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block2_2_pad (ZeroPaddin  (None, 30, 30, 128)  0          ['conv3_block2_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv3_block2_2_conv (Conv2D)   (None, 28, 28, 128)  147456      ['conv3_block2_2_pad[0][0]']     
                                                                                                  
 conv3_block2_2_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block2_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block2_2_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block2_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block2_3_conv (Conv2D)   (None, 28, 28, 512)  66048       ['conv3_block2_2_relu[0][0]']    
                                                                                                  
 conv3_block2_out (Add)         (None, 28, 28, 512)  0           ['conv3_block1_out[0][0]',       
                                                                  'conv3_block2_3_conv[0][0]']    
                                                                                                  
 conv3_block3_preact_bn (BatchN  (None, 28, 28, 512)  2048       ['conv3_block2_out[0][0]']       
 ormalization)                                                                                    
                                                                                                  
 conv3_block3_preact_relu (Acti  (None, 28, 28, 512)  0          ['conv3_block3_preact_bn[0][0]'] 
 vation)                                                                                          
                                                                                                  
 conv3_block3_1_conv (Conv2D)   (None, 28, 28, 128)  65536       ['conv3_block3_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv3_block3_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block3_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block3_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block3_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block3_2_pad (ZeroPaddin  (None, 30, 30, 128)  0          ['conv3_block3_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv3_block3_2_conv (Conv2D)   (None, 28, 28, 128)  147456      ['conv3_block3_2_pad[0][0]']     
                                                                                                  
 conv3_block3_2_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block3_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block3_2_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block3_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block3_3_conv (Conv2D)   (None, 28, 28, 512)  66048       ['conv3_block3_2_relu[0][0]']    
                                                                                                  
 conv3_block3_out (Add)         (None, 28, 28, 512)  0           ['conv3_block2_out[0][0]',       
                                                                  'conv3_block3_3_conv[0][0]']    
                                                                                                  
 conv3_block4_preact_bn (BatchN  (None, 28, 28, 512)  2048       ['conv3_block3_out[0][0]']       
 ormalization)                                                                                    
                                                                                                  
 conv3_block4_preact_relu (Acti  (None, 28, 28, 512)  0          ['conv3_block4_preact_bn[0][0]'] 
 vation)                                                                                          
                                                                                                  
 conv3_block4_1_conv (Conv2D)   (None, 28, 28, 128)  65536       ['conv3_block4_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv3_block4_1_bn (BatchNormal  (None, 28, 28, 128)  512        ['conv3_block4_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block4_1_relu (Activatio  (None, 28, 28, 128)  0          ['conv3_block4_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv3_block4_2_pad (ZeroPaddin  (None, 30, 30, 128)  0          ['conv3_block4_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv3_block4_2_conv (Conv2D)   (None, 14, 14, 128)  147456      ['conv3_block4_2_pad[0][0]']     
                                                                                                  
 conv3_block4_2_bn (BatchNormal  (None, 14, 14, 128)  512        ['conv3_block4_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv3_block4_2_relu (Activatio  (None, 14, 14, 128)  0          ['conv3_block4_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 max_pooling2d_4 (MaxPooling2D)  (None, 14, 14, 512)  0          ['conv3_block3_out[0][0]']       
                                                                                                  
 conv3_block4_3_conv (Conv2D)   (None, 14, 14, 512)  66048       ['conv3_block4_2_relu[0][0]']    
                                                                                                  
 conv3_block4_out (Add)         (None, 14, 14, 512)  0           ['max_pooling2d_4[0][0]',        
                                                                  'conv3_block4_3_conv[0][0]']    
                                                                                                  
 conv4_block1_preact_bn (BatchN  (None, 14, 14, 512)  2048       ['conv3_block4_out[0][0]']       
 ormalization)                                                                                    
                                                                                                  
 conv4_block1_preact_relu (Acti  (None, 14, 14, 512)  0          ['conv4_block1_preact_bn[0][0]'] 
 vation)                                                                                          
                                                                                                  
 conv4_block1_1_conv (Conv2D)   (None, 14, 14, 256)  131072      ['conv4_block1_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv4_block1_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block1_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block1_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block1_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block1_2_pad (ZeroPaddin  (None, 16, 16, 256)  0          ['conv4_block1_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv4_block1_2_conv (Conv2D)   (None, 14, 14, 256)  589824      ['conv4_block1_2_pad[0][0]']     
                                                                                                  
 conv4_block1_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block1_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block1_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block1_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block1_0_conv (Conv2D)   (None, 14, 14, 1024  525312      ['conv4_block1_preact_relu[0][0]'
                                )                                ]                                
                                                                                                  
 conv4_block1_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block1_2_relu[0][0]']    
                                )                                                                 
                                                                                                  
 conv4_block1_out (Add)         (None, 14, 14, 1024  0           ['conv4_block1_0_conv[0][0]',    
                                )                                 'conv4_block1_3_conv[0][0]']    
                                                                                                  
 conv4_block2_preact_bn (BatchN  (None, 14, 14, 1024  4096       ['conv4_block1_out[0][0]']       
 ormalization)                  )                                                                 
                                                                                                  
 conv4_block2_preact_relu (Acti  (None, 14, 14, 1024  0          ['conv4_block2_preact_bn[0][0]'] 
 vation)                        )                                                                 
                                                                                                  
 conv4_block2_1_conv (Conv2D)   (None, 14, 14, 256)  262144      ['conv4_block2_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv4_block2_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block2_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block2_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block2_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block2_2_pad (ZeroPaddin  (None, 16, 16, 256)  0          ['conv4_block2_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv4_block2_2_conv (Conv2D)   (None, 14, 14, 256)  589824      ['conv4_block2_2_pad[0][0]']     
                                                                                                  
 conv4_block2_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block2_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block2_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block2_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block2_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block2_2_relu[0][0]']    
                                )                                                                 
                                                                                                  
 conv4_block2_out (Add)         (None, 14, 14, 1024  0           ['conv4_block1_out[0][0]',       
                                )                                 'conv4_block2_3_conv[0][0]']    
                                                                                                  
 conv4_block3_preact_bn (BatchN  (None, 14, 14, 1024  4096       ['conv4_block2_out[0][0]']       
 ormalization)                  )                                                                 
                                                                                                  
 conv4_block3_preact_relu (Acti  (None, 14, 14, 1024  0          ['conv4_block3_preact_bn[0][0]'] 
 vation)                        )                                                                 
                                                                                                  
 conv4_block3_1_conv (Conv2D)   (None, 14, 14, 256)  262144      ['conv4_block3_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv4_block3_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block3_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block3_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block3_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block3_2_pad (ZeroPaddin  (None, 16, 16, 256)  0          ['conv4_block3_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv4_block3_2_conv (Conv2D)   (None, 14, 14, 256)  589824      ['conv4_block3_2_pad[0][0]']     
                                                                                                  
 conv4_block3_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block3_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block3_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block3_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block3_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block3_2_relu[0][0]']    
                                )                                                                 
                                                                                                  
 conv4_block3_out (Add)         (None, 14, 14, 1024  0           ['conv4_block2_out[0][0]',       
                                )                                 'conv4_block3_3_conv[0][0]']    
                                                                                                  
 conv4_block4_preact_bn (BatchN  (None, 14, 14, 1024  4096       ['conv4_block3_out[0][0]']       
 ormalization)                  )                                                                 
                                                                                                  
 conv4_block4_preact_relu (Acti  (None, 14, 14, 1024  0          ['conv4_block4_preact_bn[0][0]'] 
 vation)                        )                                                                 
                                                                                                  
 conv4_block4_1_conv (Conv2D)   (None, 14, 14, 256)  262144      ['conv4_block4_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv4_block4_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block4_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block4_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block4_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block4_2_pad (ZeroPaddin  (None, 16, 16, 256)  0          ['conv4_block4_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv4_block4_2_conv (Conv2D)   (None, 14, 14, 256)  589824      ['conv4_block4_2_pad[0][0]']     
                                                                                                  
 conv4_block4_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block4_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block4_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block4_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block4_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block4_2_relu[0][0]']    
                                )                                                                 
                                                                                                  
 conv4_block4_out (Add)         (None, 14, 14, 1024  0           ['conv4_block3_out[0][0]',       
                                )                                 'conv4_block4_3_conv[0][0]']    
                                                                                                  
 conv4_block5_preact_bn (BatchN  (None, 14, 14, 1024  4096       ['conv4_block4_out[0][0]']       
 ormalization)                  )                                                                 
                                                                                                  
 conv4_block5_preact_relu (Acti  (None, 14, 14, 1024  0          ['conv4_block5_preact_bn[0][0]'] 
 vation)                        )                                                                 
                                                                                                  
 conv4_block5_1_conv (Conv2D)   (None, 14, 14, 256)  262144      ['conv4_block5_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv4_block5_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block5_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block5_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block5_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block5_2_pad (ZeroPaddin  (None, 16, 16, 256)  0          ['conv4_block5_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv4_block5_2_conv (Conv2D)   (None, 14, 14, 256)  589824      ['conv4_block5_2_pad[0][0]']     
                                                                                                  
 conv4_block5_2_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block5_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block5_2_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block5_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block5_3_conv (Conv2D)   (None, 14, 14, 1024  263168      ['conv4_block5_2_relu[0][0]']    
                                )                                                                 
                                                                                                  
 conv4_block5_out (Add)         (None, 14, 14, 1024  0           ['conv4_block4_out[0][0]',       
                                )                                 'conv4_block5_3_conv[0][0]']    
                                                                                                  
 conv4_block6_preact_bn (BatchN  (None, 14, 14, 1024  4096       ['conv4_block5_out[0][0]']       
 ormalization)                  )                                                                 
                                                                                                  
 conv4_block6_preact_relu (Acti  (None, 14, 14, 1024  0          ['conv4_block6_preact_bn[0][0]'] 
 vation)                        )                                                                 
                                                                                                  
 conv4_block6_1_conv (Conv2D)   (None, 14, 14, 256)  262144      ['conv4_block6_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv4_block6_1_bn (BatchNormal  (None, 14, 14, 256)  1024       ['conv4_block6_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block6_1_relu (Activatio  (None, 14, 14, 256)  0          ['conv4_block6_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv4_block6_2_pad (ZeroPaddin  (None, 16, 16, 256)  0          ['conv4_block6_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv4_block6_2_conv (Conv2D)   (None, 7, 7, 256)    589824      ['conv4_block6_2_pad[0][0]']     
                                                                                                  
 conv4_block6_2_bn (BatchNormal  (None, 7, 7, 256)   1024        ['conv4_block6_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv4_block6_2_relu (Activatio  (None, 7, 7, 256)   0           ['conv4_block6_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 max_pooling2d_5 (MaxPooling2D)  (None, 7, 7, 1024)  0           ['conv4_block5_out[0][0]']       
                                                                                                  
 conv4_block6_3_conv (Conv2D)   (None, 7, 7, 1024)   263168      ['conv4_block6_2_relu[0][0]']    
                                                                                                  
 conv4_block6_out (Add)         (None, 7, 7, 1024)   0           ['max_pooling2d_5[0][0]',        
                                                                  'conv4_block6_3_conv[0][0]']    
                                                                                                  
 conv5_block1_preact_bn (BatchN  (None, 7, 7, 1024)  4096        ['conv4_block6_out[0][0]']       
 ormalization)                                                                                    
                                                                                                  
 conv5_block1_preact_relu (Acti  (None, 7, 7, 1024)  0           ['conv5_block1_preact_bn[0][0]'] 
 vation)                                                                                          
                                                                                                  
 conv5_block1_1_conv (Conv2D)   (None, 7, 7, 512)    524288      ['conv5_block1_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv5_block1_1_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block1_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block1_1_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block1_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block1_2_pad (ZeroPaddin  (None, 9, 9, 512)   0           ['conv5_block1_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv5_block1_2_conv (Conv2D)   (None, 7, 7, 512)    2359296     ['conv5_block1_2_pad[0][0]']     
                                                                                                  
 conv5_block1_2_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block1_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block1_2_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block1_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block1_0_conv (Conv2D)   (None, 7, 7, 2048)   2099200     ['conv5_block1_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv5_block1_3_conv (Conv2D)   (None, 7, 7, 2048)   1050624     ['conv5_block1_2_relu[0][0]']    
                                                                                                  
 conv5_block1_out (Add)         (None, 7, 7, 2048)   0           ['conv5_block1_0_conv[0][0]',    
                                                                  'conv5_block1_3_conv[0][0]']    
                                                                                                  
 conv5_block2_preact_bn (BatchN  (None, 7, 7, 2048)  8192        ['conv5_block1_out[0][0]']       
 ormalization)                                                                                    
                                                                                                  
 conv5_block2_preact_relu (Acti  (None, 7, 7, 2048)  0           ['conv5_block2_preact_bn[0][0]'] 
 vation)                                                                                          
                                                                                                  
 conv5_block2_1_conv (Conv2D)   (None, 7, 7, 512)    1048576     ['conv5_block2_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv5_block2_1_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block2_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block2_1_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block2_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block2_2_pad (ZeroPaddin  (None, 9, 9, 512)   0           ['conv5_block2_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv5_block2_2_conv (Conv2D)   (None, 7, 7, 512)    2359296     ['conv5_block2_2_pad[0][0]']     
                                                                                                  
 conv5_block2_2_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block2_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block2_2_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block2_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block2_3_conv (Conv2D)   (None, 7, 7, 2048)   1050624     ['conv5_block2_2_relu[0][0]']    
                                                                                                  
 conv5_block2_out (Add)         (None, 7, 7, 2048)   0           ['conv5_block1_out[0][0]',       
                                                                  'conv5_block2_3_conv[0][0]']    
                                                                                                  
 conv5_block3_preact_bn (BatchN  (None, 7, 7, 2048)  8192        ['conv5_block2_out[0][0]']       
 ormalization)                                                                                    
                                                                                                  
 conv5_block3_preact_relu (Acti  (None, 7, 7, 2048)  0           ['conv5_block3_preact_bn[0][0]'] 
 vation)                                                                                          
                                                                                                  
 conv5_block3_1_conv (Conv2D)   (None, 7, 7, 512)    1048576     ['conv5_block3_preact_relu[0][0]'
                                                                 ]                                
                                                                                                  
 conv5_block3_1_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block3_1_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block3_1_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block3_1_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block3_2_pad (ZeroPaddin  (None, 9, 9, 512)   0           ['conv5_block3_1_relu[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv5_block3_2_conv (Conv2D)   (None, 7, 7, 512)    2359296     ['conv5_block3_2_pad[0][0]']     
                                                                                                  
 conv5_block3_2_bn (BatchNormal  (None, 7, 7, 512)   2048        ['conv5_block3_2_conv[0][0]']    
 ization)                                                                                         
                                                                                                  
 conv5_block3_2_relu (Activatio  (None, 7, 7, 512)   0           ['conv5_block3_2_bn[0][0]']      
 n)                                                                                               
                                                                                                  
 conv5_block3_3_conv (Conv2D)   (None, 7, 7, 2048)   1050624     ['conv5_block3_2_relu[0][0]']    
                                                                                                  
 conv5_block3_out (Add)         (None, 7, 7, 2048)   0           ['conv5_block2_out[0][0]',       
                                                                  'conv5_block3_3_conv[0][0]']    
                                                                                                  
 post_bn (BatchNormalization)   (None, 7, 7, 2048)   8192        ['conv5_block3_out[0][0]']       
                                                                                                  
 post_relu (Activation)         (None, 7, 7, 2048)   0           ['post_bn[0][0]']                
                                                                                                  
 flatten_1 (Flatten)            (None, 100352)       0           ['post_relu[0][0]']              
                                                                                                  
 dropout_3 (Dropout)            (None, 100352)       0           ['flatten_1[0][0]']              
                                                                                                  
 dense_3 (Dense)                (None, 2048)         205522944   ['dropout_3[0][0]']              
                                                                                                  
 batch_normalization_2 (BatchNo  (None, 2048)        8192        ['dense_3[0][0]']                
 rmalization)                                                                                     
                                                                                                  
 dropout_4 (Dropout)            (None, 2048)         0           ['batch_normalization_2[0][0]']  
                                                                                                  
 dense_4 (Dense)                (None, 2048)         4196352     ['dropout_4[0][0]']              
                                                                                                  
 batch_normalization_3 (BatchNo  (None, 2048)        8192        ['dense_4[0][0]']                
 rmalization)                                                                                     
                                                                                                  
 dropout_5 (Dropout)            (None, 2048)         0           ['batch_normalization_3[0][0]']  
                                                                                                  
 dense_5 (Dense)                (None, 196)          401604      ['dropout_5[0][0]']              
                                                                                                  
==================================================================================================
Total params: 233,702,084
Trainable params: 210,129,092
Non-trainable params: 23,572,992
__________________________________________________________________________________________________


# Lisense:
MIT License

[VTMDF]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
