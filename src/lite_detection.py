import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #If the line below doesn't work, uncomment this line (make sure to comment the line below); it should help.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
'''
Numbers for "os.environ['TF_CPP_MIN_LOG_LEVEL']": 
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime
from csv import writer


def detect():

    # Import CSV file
    try:
        df = pd.read_csv("./data/mega_detection.csv", encoding="UTF-8", keep_default_na=False, na_values='') # Change encoding to UTF-16 if any errors arise
        
    except:
        df = pd.read_csv("./data/mega_detection.csv", encoding="UTF-16", keep_default_na=False, na_values='')
    
    # Get no. of rows in dataframe(df)
    df_size = len(df.index)

    # correct_predictions variable
    correct_pred = 0
    for i in range(0, df_size):
        data_array = np.array([[]], dtype=np.float32)


        sample={
            'dur': df['dur'][i],
            'proto': df['proto'][i],
            'service': df['service'][i],
            'state': df['state'][i],
            'spkts': 0,
            'dpkts': 0,
            'sbytes': 0,
            'dbytes': 0,
            'rate': 0,
            'sttl': 0,
            'dttl': 0,
            'sload': 0,
            'dload': 0,
            'sloss': 0,
            'dloss': 0,
            'sinpkt': 0,
            'dinpkt': 0,
            'sjit': 0,
            'djit': 0,
            'swin': 0,
            'stcpb': 0,
            'dtcpb': 0,
            'dwin': 0,
            'tcprtt': 0,
            'synack': 0,
            'ackdat': 0,
            'smean': 0,
            'dmean': 0,
            'trans_depth': 0,
            'response_body_len': df['response_body_len'][i],
            'ct_srv_src': df['ct_srv_src'][i],
            'ct_state_ttl': 0,
            'ct_dst_ltm': 0,
            'ct_src_dport_ltm': 0,
            'ct_dst_sport_ltm': 0,
            'ct_dst_src_ltm': df['ct_dst_src_ltm'][i],
            'is_ftp_login': df['is_ftp_login'][i],
            'ct_ftp_cmd': df['ct_ftp_cmd'][i],
            'ct_flw_http_mthd': 0,
            'ct_src_ltm': 0,
            'ct_srv_dst': df['ct_srv_dst'][i],
            'is_sm_ips_ports': df['is_sm_ips_ports'][i]
        }


        for key, value in sample.items():
            if value >= 0 and value <= 100:
                data_array = np.append(data_array, value)
            else:
                data_array = np.append(data_array, 0.0)
        

        # Create copy of data_array and reshape it to be a 3d array
        new_data_array = np.array(data_array, dtype=np.float32)
        new_data_array = new_data_array.reshape((1, 42, 1))
        

        # Create Interpreter and Allocate Tensors
        interpreter = tf.lite.Interpreter(model_path='./lite_models/model.tflite')
        interpreter.allocate_tensors()

        # Get Input and Output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the model on random input data
        #input_shape = input_details[0]['shape']


        input_data = new_data_array

        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # Get results
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_attack = np.argmax(output_data, axis=1)
        if np.float32(pred_attack[0]) == df['attack_cat'][i]:
            correct_pred += 1

    # Print out results
    print("Correct Predictions(%) : {}%".format(correct_pred * 100 / len(df['attack_cat'])))
    print("Total no. of rows: ", len(df['attack_cat']))
    print("Predicted Correctly: ", correct_pred)
    print("Wrong Predictions: ", len(df['attack_cat']) - correct_pred)

    

    # Save results to CSV file(./results/prediction_results.csv)
    pred_file_path = "./results/prediction_results.csv"

    if os.path.exists("./results/prediction_results.csv") == True:

        try:
            df2 = pd.read_csv(pred_file_path, encoding="UTF-8") # Change encoding to UTF-16 if any errors arise
        except:
            df2 = pd.read_csv(pred_file_path, encoding="UTF-16")


        now = datetime.now()
        date = [now.strftime("%d/%m/%Y %H:%M:%S")] 
        total_rows = [len(df['attack_cat'])]
        correct_preds = [correct_pred] 
        wrong_preds = [len(df['attack_cat']) - correct_pred] 
        success_perc = [correct_pred * 100 / len(df['attack_cat'])] 

        data = {
            'date': date,
            'total rows': total_rows,
            'correct predictions': correct_preds,
            'wrong predictions': wrong_preds,
            'success percentage': success_perc
        }

        new_df = pd.DataFrame(data)

        new_df.to_csv(pred_file_path, mode='a', index=False, header=False)
    else: 
        now = datetime.now()
        date = [now.strftime("%d/%m/%Y %H:%M:%S")] 
        total_rows = [len(df['attack_cat'])]
        correct_preds = [correct_pred] 
        wrong_preds = [len(df['attack_cat']) - correct_pred] 
        success_perc = [correct_pred * 100 / len(df['attack_cat'])] 

        # dictionary of lists  
        dict = {'date': date, 'total rows': total_rows, 'correct predictions': correct_preds, 'wrong predictions': wrong_preds, 'success percentage': success_perc}  

        df3 = pd.DataFrame(dict) 

        # saving the dataframe 
        df3.to_csv(pred_file_path, index=False) 



detect()
