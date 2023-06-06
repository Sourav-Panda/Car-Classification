# Vechile type and make detection ussing transfer learning and flask deployment.

Objective of this project was to train a nural network model ussing transfer learning and deploy it in aws creating a flask app.

# Installations
Python libraries:
1. Numpy
2. pandas
3. tensorflow version 1.12.0
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
refer arch.txt

# Lisense:
MIT License

[VTMDF]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
