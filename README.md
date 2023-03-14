# RNN-IDS
This is a project I developed during an ERASMUS+ Internship.
For this project I had to develop an Intrusion Detection Model in Tensorflow and then convert it to Tensorflow Lite.

**Note: This was my first time developing a Machine Learning model and training it so its accuracy is not great, neither is the logic.**


## Create Dataset for Lite Testing
1. **Download and Install TShark**
   Follow the instructions [here](https://tshark.dev/setup/install/).

2. **Change TShark Path in Python script**
   In `pcap2csv.py` change the TShark path in the `tshark_path` variable and `subprocess.call` method.

3. **Change the PCAP/CSV folders location**
   In `pcap2csv.py` change the following variable paths: `pcap_path` and `csv_path`.

4. **Run pcap2csv.py**
   This script will, automatically, convert all the PCAP files to one single CSV file (dataset) using TShark. Please allow the script to finish, as canceling it half-way will fill your CSV folder with a CSV file per PCAP file.

5. **Run detect_processing.py**
   After having the `mega_detection.csv` file in your CSV folder, run detect_processing to perform Pre-Processing on the data.

6. **Run detect_processing.py again**
   By running this script again, you will delete unnecessary rows and make sure all the data has been pre-processed. If you don't run it again, you may not be able to test the Lite Model using this Dataset.


## Usage - Google Colab
1. **Train the Model** 
    Run [model_training.ipynb](https://gitlab.com/TisaLabs/ids/ids-lite/-/blob/feature/diogo/src/Colab%20Notebooks/model_training.ipynb). If you wish, you may change the parameters, or just leave the default ones.
    This step will train a model using the set parameters and quantize the model.

2. **Train the Quantized Model (Optional)**
   Run [quantized_training.ipynb](https://gitlab.com/TisaLabs/ids/ids-lite/-/blob/feature/diogo/src/Colab%20Notebooks/quantized_training.ipynb). Change the parameters to be equal to those in the model training script. 

3. **Convert Models to Lite**
   To convert the models to Lite, run [convert_lite.ipynb](https://gitlab.com/TisaLabs/ids/ids-lite/-/blob/feature/diogo/src/Colab%20Notebooks/convert_lite.ipynb). This step will also convert the Quantized Model, if created in Step 2.

4. **Test Lite Models**
   When both models are converted, you can test them in [lite_detection.ipynb](https://gitlab.com/TisaLabs/ids/ids-lite/-/blob/feature/diogo/src/Colab%20Notebooks/lite_detection.ipynb).
   The script will ask you which one you want to use(1 for Lite, 2 for Quantized Lite). After inputting the model, it will start testing and display the results.
 


## Project structure
### Required packages:
Packages with the version used (tensorflow-gpu is only mandatory for gpu training):
* `scikit-learn==0.21.2` 
* `numpy==1.16.4`
* `pandas==0.25.0`
* `Keras==2.2.4`
* `tensorflow==1.14.0`
* `tensorboard==1.14.0`
* `tensorflow-gpu==1.14.0`


## Data
The project currently uses the UBSW-NB15 dataset for training:
* [UBSW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/) (2015).

