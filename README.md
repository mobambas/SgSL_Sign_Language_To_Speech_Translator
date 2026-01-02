## SgSL_Sign_Language_To_Speech_Translator (Hand Gesture Recognition)

# Project Description
This project focuses on recognizing American Sign Language (ASL) hand gestures using Convolutional Neural Networks (CNNs). The project leverages a dataset of skeletal hand images representing the ASL alphabet (A to Z) to train and evaluate a deep learning model for accurate gesture recognition.

# Problem Addressed
Communication barriers exist for individuals who rely on ASL for communication, especially in environments where others do not understand ASL. This project aims to bridge this gap by providing a tool that translates ASL hand gestures into corresponding letters of the alphabet, thereby facilitating better communication between ASL users and non-users.

# Solution
We developed a simple CNN-based model trained on the AtoZ_3.1 dataset, which contains hundreds of skeletal hand images for each ASL letter. The model can predict the letter represented by a given hand gesture image, whilst taking into account the subtle local nuances in individual hand gestures for the same letter. This was done by consulting actual mute and deaf people in our contacts who use ASL to communicate on a daily basis. Additionally, we also provide our original scripts for the initial data collection and prediction process, including binary hand data collection.

# Impact
This project can significantly enhance communication for ASL users, particularly in educational, social, and professional settings. By automating the recognition of ASL gestures, it provides a stepping stone toward more inclusive and accessible communication technologies.

# Setup Instructions
_1. Clone the Repository:_
```
git clone <https://github.com/mobambas/ASL_Sign_Language_To_Speech_Translator>
cd <ASL_Sign_Language_To_Speech_Translator>
```
_2. Install the Required Libraries. Ensure you have Python 3.12.4 installed first, then install the required libraries and frameworks using the below command:_
```
pip install -r requirements.txt
```
_3. Prepare the Dataset:_

Ensure the AtoZ_3.1 dataset is available in the project directory. The dataset should be structured with folders for each letter, containing the respective skeletal hand images.

_4. Model File:_

The trained model file 'cnn8grps_rad1_model.h5' should also be placed in the same project directory.

_5. File paths:_

Lastly, some of the file paths used might not be relative and instead absolute. Do remember to replace these with the correct file paths local to your system (the reason our team couldn't implement relative paths for jpg files is because it led to some issues with our python venv. We do apologise for the inconvenience). The list of paths to be replaced is as follows:
```
- "C:\\Users\\Arush\\ASL_Sign_Language_To_Speech_Translator\\CUSTOM_OBJECT_DETECTION_MODEL\\AtoZ_3.1\\A\\"
- "C:\\Users\\Arush\\ASL_Sign_Language_To_Speech_Translator\\CUSTOM_OBJECT_DETECTION_MODEL\\white.jpg"
- "C:\\Users\\Arush\\ASL_Sign_Language_To_Speech_Translator\\CUSTOM_OBJECT_DETECTION_MODEL\\AtoZ_3.1\\"
```
Note: Do remember to use the double-backslash in your file paths as if not python will treat a single backslash like an escape character.

# File Overview
1. cnn8grps_rad1_model.h5: Trained CNN model for ASL hand gesture recognition.
2. data_collection_binary.py: Script for collecting binary hand data.
3. data_collection_final.py: Script for collecting final hand gesture data.
4. final_pred.py: Final project script for making predictions using the trained model and includes a user-friendly interface, including additional auto-complete and sentence-forming features. Also has speech synthesis capabilities. 
5. prediction_wo_gui.py: Initial script for making single-character predictions without a GUI.

# Conclusion
This project aims to make ASL gesture recognition more accessible and accurate, aiding in communication for ASL users. With future enhancements, this tool can be integrated into various applications, promoting inclusivity and understanding.




