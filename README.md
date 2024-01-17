
# Dermatological lesion prediction
## Skin disease dermatoscopic classification using convolutional networks on unbalanced dataset

The skin is the largest organ in the body and the first barrier for defending our inner organs against aggression, besides helping the body regulate its temperature. Considering the importance of the skin for human beings, it is necessary to examine skin lesions. Because of that, dermatologists use dermatoscopy to illuminate and magnify the area to be examined to monitor any present injury.

In recent decades, enormous advances in the AI medical field have experienced significant improvements due to the use of deep networks. They can diagnose with reliability as well as speed.

The following project presents the use of deep learning algorithms in order to classify eight different diagnoses, such as melanoma, basal cell carcinoma, vascular lesion, and other lesions. In this study, we design and test two of today’s most commonly used convolutional networks in the context of skin lesion classification, using the 2019 ISIC dataset. This dataset presented quality challenges, including resolution variations and very significant inter-class imbalances. We implemented two strategies to solve the dataset problem: an under-sampling and an over-sampling of the minority classes. We use Synthetic minorities (SMOTE technique) to enrich the minority classes, resulting in a 20 % increase in the total number of images processed. Furthermore, we employ model training transfer learning techniques using EfficientNet B0 and ResNet50 as a model base, demonstrating significant improvements in classification metrics. In particular, ResNet50 showed higher performance when trained on the SMOTE-enricheddataset, and this can be attributed to its profound architecture benefiting from a larger dataset.

<image src="images/Introduccion/skin_lesion_sample.png" alt="Skin lesions sample">


This research contributes to a new understanding of skin lesion classification through data preprocessing, transfer learning, and model selection, paving the way for future enhancements in dermatological image analysis and early diagnosis. Finally, it also contributes exploring the use of a statistic technique like SMOTE to solve an imbalanced dataset challenge. 

**Table of contents:**

* [Introduction](#readme.rd)
* [Master's thesis](#instalación)
* [Jupyter Notebooks (web version)](#uso)
* [Video Presentation] (https://1drv.ms/v/s!AsEQ8KcFiwGBhMAj2bCsQVLPDtOZZA?e=DMKwjd)https://1drv.ms/v/s!AsEQ8KcFiwGBhMAj2bCsQVLPDtOZZA?e=DMKwjd)
* [License](#licencia)
