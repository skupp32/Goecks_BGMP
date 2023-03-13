# Challenges and Next Steps:

## Class Imbalance and Sample Size:

Class imbalance alongside smaller sample size is likely introducing sampling bias. The download manifesto consists of 875 raw images from TCGA. The distribution is shown below:

<p align="center">
<img width="393" alt="Screen Shot 2023-03-12 at 3 37 34 PM" src="https://user-images.githubusercontent.com/106117735/224578084-2e6d6978-7f4c-4f83-9d52-39c0a0975b44.png">
</p>

The initial difference between n samples for minority and majority subclasses is significant across histological and PAM50 subtypes. We attempted to minimize this effect by tiling the raw image data, in order to inflate samples per raw image. However, the imbalance persists throughout preprocessing. 

Therefore, we’ve implemented a downsampling method to ensure that each subtype class has the same amount of data. This method involves taking a random sample of each subtype based on the number of samples in the minority class, and using this as input. While this guaranteed that the classes would be balanced, there was significant data loss due to the minority class being much less than the average sample size per class. 

Thus, we’ve adjusted the scope of the model from classifying histological and PAM50 subtypes (10 classes), to classifying PAM50 subtypes only. (5 Classes) Even so, there is an insignificant performance increase. (See latest version of ResNet script results) In future efforts, we suggest that initial manifests from TCGA consist of more data, and class balance in mind. 

## Quality Trimming:

The goal of quality trimming raw image data is to remove unnecessary noise which could introduce bias to the model. This is implemented within the tiling functions in the preprocessing script. However, a high quality tile is currently defined by the proportion of white pixels within it. This is a necessary step, but doesn’t quite address the goal of highlighting important information within our data. Given the assumption that the most impactful data within each whole slide image is the proportion consisting of cancerous cells, we’ve only enhanced the data to focus on the proportion of the slide which is stained with H&E. 

In an attempt to improve upon this, we adjusted the stain normalization function in our preprocessing script to separate hematoxylin components from the image, as hematoxylin stains the nuclear and ribosomal components of cells. In testing model performance with this modification, we believe it negatively impacted the ability for the model to distinguish subtypes. Due to this, it was not implemented in the final model. However, we believe stain separation should be investigated further in the future. While the performance of our model depends on the credibility of its annotated input data, future efforts should work to ensure that representations of each subclass are clearly defined. Perhaps by first identifying cancerous cells within each whole slide image, and cropping to this section of the image before preprocessing.  

