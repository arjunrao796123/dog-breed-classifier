# Dog-breed-classifier

Udacity dog breed project
Build a pipeline to process real-world, user-supplied images.
Given an image of a dog, the algorithm will identify an estimate of the canine’s breed. If supplied an image of a human face, the code will identify the resembling dog breed.

Detect Humans
Assess the Human Face Detector The submission returns the percentage of the first 100 images in the dog and human face datasets that include a detected, human face.

Detect Dogs
Use a pre-trained VGG16 Net to find the predicted class for a given image: dog_detector function returns True if a dog is detected in an image and False if not.

We use various neworks ike Xception model, Inception model, VGG19, Resnet 50 to compare which model performs the best.

You can view the blog post at https://medium.com/@arao_81907/dog-breed-identifier-udacity-nano-degree-project-d00f6126260


The dog dataset can be downloaded at 
[Dog Dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip "Dog Data")


The human dataset can be downloaded at 
[Human Dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip "Human Data")

If you want to use a GPU to run this code, then you can use Google colab or an aws instance.

# Best models
The best model for me was the xception model, which gave me an accuracy of 85.7%


# Acknowledgements
I would like to thank Udacity for this opportunity.
