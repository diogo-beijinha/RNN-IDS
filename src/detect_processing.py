from codecs import ignore_errors
from enum import auto
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
import csv


def data_processing():

    # Script for creating Dataset for Lite Model Testing
    try:
        df = pd.read_csv("./data/mega_detection.csv", encoding="UTF-8", na_values='0', keep_default_na=False)
        
    except:
        df = pd.read_csv("./data/mega_detection.csv", encoding="UTF-16", na_values='0', keep_default_na=False)

    try:
        # Change header names
        print("Changing header names")
        df.rename(columns={'frame.time_delta': 'dur', 'frame.protocols': 'proto', 'frame.len': 'response_body_len', 'ftp.command-frame': 'ct_ftp_cmd', 'tcp.analysis.ack_rtt': 'tcprtt'}, inplace=True)
        print("Header names modified with success")
    except:
        pass

    # Protocols used in training
    proto_values = ['tcp', 'udp', 'arp', 'ospf', 'icmp', 'igmp', 'rtp', 'ddp',
                'ipv6-frag', 'cftp', 'wsn', 'pvp', 'wb-expak', 'mtp',
                'pri-enc', 'sat-mon', 'cphb', 'sun-nd', 'iso-ip', 'xtp', 'il',
                'unas', 'mfe-nsp', '3pc', 'ipv6-route', 'idrp', 'bna', 'swipe',
                'kryptolan', 'cpnx', 'rsvp', 'wb-mon', 'vmtp', 'ib', 'dgp',
                'eigrp', 'ax.25', 'gmtp', 'pnni', 'sep', 'pgm', 'idpr-cmtp',
                'zero', 'rvd', 'mobile', 'narp', 'fc', 'pipe', 'ipcomp',
                'ipv6-no', 'sat-expak', 'ipv6-opts', 'snp', 'ipcv',
                'br-sat-mon', 'ttp', 'tcf', 'nsfnet-igp', 'sprite-rpc',
                'aes-sp3-d', 'sccopmce', 'sctp', 'qnx', 'scps', 'etherip',
                'aris', 'pim', 'compaq-peer', 'vrrp', 'iatp', 'stp',
                'l2tp', 'srp', 'sm', 'isis', 'smp', 'fire', 'ptp', 'crtp',
                'sps', 'merit-inp', 'idpr', 'skip', 'any', 'larp', 'ipip',
                'micp', 'encap', 'ifmp', 'tp++', 'a/n', 'ipv6', 'i-nlsp',
                'ipx-n-ip', 'sdrp', 'tlsp', 'gre', 'mhrp', 'ddx', 'ippc',
                'visa', 'secure-vmtp', 'uti', 'vines', 'crudp', 'iplt',
                'ggp', 'ip', 'ipnip', 'st2', 'argus', 'bbn-rcc', 'egp',
                'emcon', 'igp', 'nvp', 'pup', 'xnet', 'chaos', 'mux', 'dcn',
                'hmp', 'prm', 'trunk-1', 'xns-idp', 'leaf-1', 'leaf-2', 'rdp',
                'irtp', 'iso-tp4', 'netblt', 'trunk-2', 'cbt']

    # States used in training
    state_values = ['FIN', 'INT', 'CON', 'ECO', 'REQ', 'RST', 'PAR', 'URN', 'no',
                    'ACC', 'CLO']

    # Services used in training
    service_values = ['-', 'ftp', 'smtp', 'snmp', 'http', 'ftp-data',
                    'dns', 'ssh', 'radius', 'pop3', 'dhcp', 'ssl', 'irc']

    # Attack Categories used in training
    attack_cat_values = ['Normal', 'Backdoor', 'Analysis', 'Fuzzers', 'Shellcode',
                        'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic']

# Create Service feature
    if 'service' not in df.columns.values:
        print("Creating Services feature in CSV file")
        df['service'] = ""
    else:
        pass

#Process Protocol Values
    print("Processing Protocols")
    try:
        for i in range(0, len(df['frame.protocols'])):
            if type(df['frame.protocols'][i]) == str:
                if "eth" in df['frame.protocols'][i]:
                    for service in service_values:
                        if service in df['frame.protocols'][i]:
                            df['service'][i] = service
                            break
                    for protocol in proto_values:
                        if protocol in df['frame.protocols'][i]:
                            df['frame.protocols'][i] = protocol
                            break
            else:
                pass
    except:
        for i in range(0, len(df['proto'])):
            if type(df['proto'][i]) == str:
                if "eth" in df['proto'][i]:
                    for service in service_values:
                        if service in df['proto'][i]:
                            df['service'][i] = service
                            break
                    for protocol in proto_values:
                        if protocol in df['proto'][i]:
                            df['proto'][i] = protocol
                            break
                    else:
                        if type(df['proto'][i]) == str:
                            df['proto'][i] = "dropthisrow"
            else:
                pass
    
    print("Protocols Processed")

    # Create Service Feature
    for i in range(0, len(df['service'])):
        if not df['service'][i]:
            df['service'][i] = "-"


    # Create is_ftp_login feature
    if 'is_ftp_login' not in df.columns.values:
        print("Creating 'is_ftp_login' feature in CSV file")
        df['is_ftp_login'] = ""
    else:
        pass

    # Process is_ftp_login
    if df['ftp.response.code'][i] == 230:
        df['is_ftp_login'][i] = 1
    else:
        df['is_ftp_login'][i] = 0
    
    try:
        for i in range(0, len(df['frame.protocols'])):
            if df['frame.protocols'][i] == "ftp" and df['tcp.srcport'][i] == 21 or df['tcp.dstport'][i] == 21:
                df['is_ftp_login'][i] = 1
            else:
                df['is_ftp_login'][i] = 0
    except:
        for i in range(0, len(df['proto'])):
            if df['proto'][i] == "ftp" and df['tcp.srcport'][i] == 21 or df['tcp.dstport'][i] == 21:
                df['is_ftp_login'][i] = 1
            else:
                df['is_ftp_login'][i] = 0
    

    # create ct_srv_src feature
    print("Creating 'ct_srv_src' feature in CSV file")
    if 'ct_srv_src' not in df.columns.values:
        df['ct_srv_src'] = 0
    else:
        pass
    
    # Process ct_srv_src
    print("Inserting data in 'ct_srv_src' feature in CSV file")
    for i in range(0, len(df['ct_srv_src'])):
        curr_service_ip = str(df['service'][i]) + str(df['ip.src'][i])
        curr_srv_ip_counter = 0
        array_equal = []
        for j in range(i, 99):
            if df['ct_srv_src'][j] < 2:
                j_service_ip = str(df['service'][j]) + str(df['ip.src'][j])
                if j_service_ip == curr_service_ip:
                    curr_srv_ip_counter += 1
                    array_equal.append(j)
        for n in array_equal:
            df['ct_srv_src'][n] = curr_srv_ip_counter
    
    for i in range(0, len(df['ct_srv_src'])):
        if df['ct_srv_src'][i] == 0:
            df['ct_srv_src'][i] = 1
        


    # create ct_dst_src_ltm feature
    print("Creating 'ct_dst_src_ltm' feature in CSV file")
    if 'ct_dst_src_ltm' not in df.columns.values:
        df['ct_dst_src_ltm'] = 0
    else:
        pass
    
    # Process ct_dst_src_ltm
    print("Inserting data in 'ct_dst_src_ltm' feature in CSV file")
    for i in range(0, len(df['ct_dst_src_ltm'])):
        curr_src_dst = str(df['ip.dst'][i]) + str(df['ip.src'][i])
        curr_src_dst_counter = 0
        array_equal = []
        for j in range(i, 99):
            if df['ct_dst_src_ltm'][j] < 2:
                j_src_dst = str(df['ip.dst'][j]) + str(df['ip.src'][j])
                if j_src_dst == curr_src_dst:
                    curr_src_dst_counter += 1
                    array_equal.append(j)
        for n in array_equal:
            df['ct_dst_src_ltm'][n] = curr_src_dst_counter
    
    for i in range(0, len(df['ct_dst_src_ltm'])):
        if df['ct_dst_src_ltm'][i] == 0:
            df['ct_dst_src_ltm'][i] = 1



    # create ct_srv_dst feature
    print("Creating 'ct_srv_dst' feature in CSV file")
    if 'ct_srv_dst' not in df.columns.values:
        df['ct_srv_dst'] = 0
    else:
        pass

    # Process ct_srv_dst_dict
    print("Inserting data in 'ct_srv_dst' feature in CSV file")
    for i in range(0, len(df['ct_srv_dst'])):
        curr_ip_srv = str(df['service'][i]) + str(df['ip.dst'][i])
        curr_ip_srv_counter = 0
        array_equal = []
        for j in range(i, 99):
            if df['ct_srv_dst'][j] < 2:
                j_ip_srv = str(df['service'][j]) + str(df['ip.dst'][j])
                if j_ip_srv == curr_ip_srv:
                    curr_ip_srv_counter += 1
                    array_equal.append(j)
        for n in array_equal:
            df['ct_srv_dst'][n] = curr_ip_srv_counter
    
    for i in range(0, len(df['ct_srv_dst'])):
        if df['ct_srv_dst'][i] == 0:
            df['ct_srv_dst'][i] = 1
        

    # Create state feature
    print("Creating State feature in CSV file")
    if 'state' not in df.columns.values:
        df['state'] = ""
    else:
        pass
    
    # Process state feature 
    # Only have 3 states
    try:
        for i in range(0, len(df['state'])):
            if df['tcp.connection.fin'][i] == "1.0" or df['tcp.flags.fin'][i] == "1.0":
                df['state'][i] = 'FIN'
            if df['tcp.connection.rst'][i] == "1.0":
                df['state'][i] = 'RST'
            if df['http.request'][i] == "1.0":
                df['state'][i] = 'REQ'
    except:
        pass


    # Create is_sm_ips_ports feature
    print("Creating 'is_sm_ips_ports' feature in CSV file")
    if 'is_sm_ips_ports' not in df.columns.values:
        df['is_sm_ips_ports'] = ""
    else:
        pass

    # Process is_sm_ips_ports feature
    for i in range(0, len(df['ip.src'])):
        if df['ip.src'][i] == df['ip.dst'][i] and df['tcp.srcport'][i] == df['tcp.dstport'][i]:
            if df['ip.src'][i] == "" or df['ip.src'][i] == np.NaN:
                df['is_sm_ips_ports'][i] = 0
            else:
                df['is_sm_ips_ports'][i] = 1
        else:
            df['is_sm_ips_ports'][i] = 0


    # Create attack_cat feature
    print("Creating 'attack_cat' feature in CSV file")
    if 'attack_cat' not in df.columns.values:
        df['attack_cat'] = ''
    else:
        pass


    # Create sbytes feature
    print("Creating 'sbytes' feature in CSV file")
    if 'sbytes' not in df.columns.values:
        df['sbytes'] = ""
    else:
        pass

    # Create dbytes feature
    print("Creating 'dbytes' feature in CSV file")
    if 'dbytes' not in df.columns.values:
        df['dbytes'] = ""
    else:
        pass

    # Process 'sbytes' and 'dbytes'
    for i in range(0, len(df['response_body_len'])):
        try:
            if df['ip.src'][i] == df['ip.dst'][i+1]:
                if df['ip.dst'][i] == df['ip.src'][i+1]:
                    if df['sbytes'][i] == "" and df['dbytes'][i] == "":
                        df['sbytes'][i] = df['response_body_len'][i]
                        df['dbytes'][i] = df['response_body_len'][i+1]
                        df['sbytes'][i+1] = df['response_body_len'][i]
                        df['dbytes'][i+1] = df['response_body_len'][i+1]
            if df['sbytes'][i] == "":
                df['sbytes'][i] = 0
            if df['dbytes'][i] == "":
                df['dbytes'][i] = 0
        except:
            pass


    # Convert "arp.opcode" to float
    for i in range(0, len(df['arp.opcode'])):
        if type(df['arp.opcode'][i]) == str:
            if df['arp.opcode'][i] == '':
                df['arp.opcode'][i] = float(0.0)
            else:
                df['arp.opcode'][i] = float(df['arp.opcode'][i])

    # Process attack_cat
    # To add more attack categories, you need to add more conditions here
    print("Categorizing attacks..")
    for i in range(0, len(df['attack_cat'])):
        if df['arp.dst.hw_mac'][i] == "00:00:00:00:00:00":
            df['attack_cat'][i] = 'Reconnaissance'
        if df['icmp.type'][i] == 3 and df['icmp.code'][i] == 2:
            df['attack_cat'][i] = 'Reconnaissance' 
        if df['icmp.type'][i] == 8 or df['icmp.type'][i] == 0:
            df['attack_cat'][i] = 'Reconnaissance'
        if df['tcp.dstport'][i] == 7:
            df['attack_cat'][i] = 'Reconnaissance'
        if df['udp.dstport'][i] == 7:
            df['attack_cat'][i] = 'Reconnaissance'
        if df['tcp.flags.syn'][i] == 1 and df['tcp.flags.ack'][i] == 0 and df['tcp.window_size'][i] <= 1024:
            df['attack_cat'][i] = 'Analysis'
        if df['tcp.flags.syn'][i] == 1 and df['tcp.flags.ack'][i] == 0 and df['tcp.window_size'][i] > 1024:
            df['attack_cat'][i] = 'Analysis'
        if df['tcp.flags'][i] == 0:
            df['attack_cat'][i] = 'Analysis'
        if df['tcp.flags'][i] == 0x001:
            df['attack_cat'][i] = 'Analysis'
        if df['tcp.flags.fin'][i] == 1 and df['tcp.flags.push'][i] == 1 and df['tcp.flags.urg'][i] == 1:
            df['attack_cat'][i] = 'Analysis'
        if df['icmp.type'][i] == 3 and df['icmp.code'][i] == 3:
            df['attack_cat'][i] = 'Analysis'
        if df['arp.duplicate-address-detected'][i] or df['arp.duplicate-address-frame'][i]:
            df['attack_cat'][i] = 'Exploits'
        if df['icmp'][i] == 1 and df['data.len'][i] > 48:
            df['attack_cat'][i] = 'DoS'
        if df['dtp'][i] or df['vlan.too_many_tags'][i]:
            df['attack_cat'][i] = 'Exploits'
        if df['tcp.analysis.lost_segment'][i] or df['tcp.analysis.retransmission'][i]:
            df['attack_cat'][i] = 'Exploits'
        if df['tcp.flags.syn'][i] == 1 and df['tcp.flags.ack'][i] == 0:
            df['attack_cat'][i] = 'DoS'
        if df['icmp'][i] == 1:
            df['attack_cat'][i] = 'Exploits'
        if df['tcp.flags.syn'][i] == 1 or df['tcp.flags.reset'][i] == 1:
            df['attack_cat'][i] = 'Reconnaissance'
        if df['ftp.response.code'][i] == 530:
            df['attack_cat'][i] = 'Exploits'
        if (df['arp.opcode'][i]) > 0:
            df['attack_cat'][i] = 'Exploits' 

    if "id" in df.columns.values:
        k = 0
        for i in range(0, len(df['id'])):
            # k += 2
            if i + 2 < len(df['id']):
                if df['proto'][i] == df['proto'][i+1] and df['proto'][i] == df['proto'][i+2] or df['tcp.dstport'][i] == df['tcp.dstport'][i+1] and df['tcp.dstport'][i] == df['tcp.dstport'][i+2]:
                    if df['ip.src'][i] != "" and df['ip.dst'][i] != "":
                        if df['ip.src'][i] == df['ip.src'][i+1] and df['ip.src'][i] == df['ip.src'][i+2]:
                            if df['ip.dst'][i] == df['ip.dst'][i+1] and df['ip.dst'][i] == df['ip.dst'][i+1]:
                                df['attack_cat'][i] = "DoS"
                                df['attack_cat'][i+1] = "DoS"
                                df['attack_cat'][i+2] = "DoS"
        
    print("Attack Categorizing Done")



    # Replace attack_cat with int values
    print("Converting 'attack_cat' values to int")
    for i in range(0, len(df['attack_cat'])):
        if df['attack_cat'][i] == "Backdoors":
            df['attack_cat'][i] = 'Backdoor'
            pass
        if type(df['attack_cat'][i]) == str and df['attack_cat'][i].isalpha() == True and df['attack_cat'][i] != "Backdoors":
            df['attack_cat'][i] = df['attack_cat'][i].strip()
        for value in attack_cat_values:
                if df['attack_cat'][i] == value:
                        att_str = df['attack_cat'][i]
                        df['attack_cat'][i] = attack_cat_values.index(att_str)
                        df['attack_cat'][i] = np.float32(df['attack_cat'][i])
            


    # Replace protocol string values with their respective int values
    print("Converting Protocol values to int")
    try:
        for i in range(0, len(df['proto'])):
            if type(df['proto'][i]) != np.int64:
                for value in proto_values:
                    if df['proto'][i] == value:
                            proto_str = df['proto'][i]
                            df['proto'][i] = proto_values.index(proto_str)
    except:
        for i in range(0, len(df['frame.protocols'])):
            if type(df['frame.protocols'][i]) != np.float32:
                for value in proto_values:
                    if df['frame.protocols'][i] == value:
                            proto_str = df['frame.protocols'][i]
                            df['frame.protocols'][i] = proto_values.index(proto_str)


    # Replace service with their int correspondences
    print("Converting Service values to int")
    for i in range(0, len(df['service'])):
            if type(df['service'][i]) != np.int64:
                for value in service_values:
                    if df['service'][i] == value:
                            service_str = df['service'][i]
                            df['service'][i] = service_values.index(service_str)


    # Replace state with their int correspondences
    print("Converting State values to int")
    for i in range(0, len(df['state'])):
            if type(df['state'][i]) != np.int64:
                for value in state_values:
                    if df['state'][i] == value:
                            state_str = df['state'][i]
                            df['state'][i] = state_values.index(state_str)



    # Process 'dttl' and 'sttl' 
    for i in range(0, len(df['ip.ttl'])):
        try:
            if df['ip.src'][i] == df['ip.dst'][i+1]:
                if df['ip.dst'][i] == df['ip.src'][i+1]:
                    if df['dttl'][i] == "" and df['sttl'][i] == "":
                        if df['dns.resp.ttl'][i]:
                            df['sttl'][i] = df['dns.resp.ttl'][i]
                            df['dttl'][i] = df['dns.resp.ttl'][i+1]
                            df['sttl'][i+1] = df['dns.resp.ttl'][i]
                            df['dttl'][i+1] = df['dns.resp.ttl'][i+1]
                        else:
                            df['sttl'][i] = df['ip.ttl'][i]
                            df['dttl'][i] = df['ip.ttl'][i+1]
                            df['sttl'][i+1] = df['ip.ttl'][i]
                            df['dttl'][i+1] = df['ip.ttl'][i+1]
            if df['sttl'][i] == "":
                df['sttl'][i] = 0
            if df['dttl'][i] == "":
                df['dttl'][i] = 0
        except:
            print("Creating 'dttl' and 'sttl' features in CSV file")
            if 'dttl' not in df.columns.values:
                df['dttl'] = ""
                df['sttl'] = ""
            else:
                pass

    # Process 'spkts' and 'dpkts' 
    for i in range(0, len(df['response_body_len'])):
        try:
            if df['ip.src'][i] == df['ip.dst'][i+1]:
                if df['ip.dst'][i] == df['ip.src'][i+1]:
                    if df['spkts'][i] == 0 and df['dpkts'][i] == 0:
                        df['spkts'][i] = df['response_body_len'][i]
                        df['dpkts'][i] = df['response_body_len'][i+1]
                        df['spkts'][i+1] = df['response_body_len'][i]
                        df['dpkts'][i+1] = df['response_body_len'][i+1]
            if df['dpkts'][i] == "":
                df['dpkts'][i] = 0
            if df['spkts'][i] == "":
                df['spkts'][i] = 0
        except:
            print("Creating 'spkts' and 'dpkts' features in CSV file")
            if 'spkts' not in df.columns.values:
                df['spkts'] = 0
                df['dpkts'] = 0
            else:
                pass

    # Process 'stcpb' and 'dtcpb' 
    for i in range(0, len(df['response_body_len'])):
        try:
            if df['tcp.connection.syn'][i] == "1.0":
                df['stcpb'][i] = df['tcp.seq'][i]
            else:
                df['dtcpb'][i] = df['tcp.seq'][i]

            if df['dtcpb'][i] == "":
                df['dtcpb'][i] = 0
            if df['stcpb'][i] == "":
                df['stcpb'][i] = 0
        except:
            print("Creating 'stcpb' and 'dtcpb' features in CSV file")
            if 'stcpb' not in df.columns.values:
                df['stcpb'] = 0
                df['dtcpb'] = 0

    # Process 'sintpkt' and 'dintpkt' 
    for i in range(0, len(df['frame.time_relative'])):
        try:
            if df['ip.src'][i] == df['ip.dst'][i+1]:
                if df['ip.dst'][i] == df['ip.src'][i+1]:
                    if df['sintpkt'][i] == 0 and df['dintpkt'][i] == 0:
                        df['sintpkt'][i] = df['frame.time_relative'][i]
                        df['dintpkt'][i] = df['frame.time_relative'][i+1]
                        df['sintpkt'][i+1] = df['frame.time_relative'][i]
                        df['dintpkt'][i+1] = df['frame.time_relative'][i+1]
            if df['dintpkt'][i] == "":
                df['dintpkt'][i] = 0
            if df['sintpkt'][i] == "":
                df['sintpkt'][i] = 0
        except:
            if 'sintpkt' not in df.columns.values:
                df['sintpkt'] = 0
                df['dintpkt'] = 0
            else:
                pass
    
    # Process 'sloss' and 'dloss' 
    for i in range(0, len(df['tcp.analysis.lost_segment'])):
        try:
            if df['tcp.analysis.lost_segment'][i] == "1.0" or df['tcp.analysis.ack_lost_segment'][i] == "1.0" or df['tcp.analysis.retransmission'][i] == "1.0" or df['tcp.analysis.fast_retransmission'][i] == "1.0" or df['tcp.analysis.spurious_retransmission'][i] == "1.0": # Lost fragments
                for t in range(i-1, 0, -1):
                    if df['tcp.srcport'][t] == df['tcp.srcport'][i]:
                        if df['tcp.dstport'][t] == df['tcp.dstport'][i]:
                            if df['tcp.analysis.lost_segment'][t] != "1.0" and df['tcp.analysis.ack_lost_segment'][t] != "1.0" and df['tcp.analysis.retransmission'][t] != "1.0" and df['tcp.analysis.fast_retransmission'][t] != "1.0" and df['tcp.analysis.spurious_retransmission'][t] != "1.0":
                                if df['tcp.connection.syn'][t] == "1.0":
                                    df['sloss'][t] = df['response_body_len'][i] # Only for testing! Not correct!
                                else:
                                    df['dloss'][t] = df['response_body_len'][i] # Only for testing! Not correct!
                                break
                                
            # If empty, replace with 0
            if df['dloss'][i] == "":
                df['dloss'][i] = 0
            if df['sloss'][i] == "":
                df['sloss'][i] = 0
        except:
            # If columns don't exist, create them
            if 'dloss' not in df.columns.values:
                df['dloss'] = 0
                df['sloss'] = 0
            else:
                pass
        
       
    # Replace 'NaN' values with 0
    df = df.fillna(0)
    print("Filled empty fields with '0'")
    
    
    # Convert 'service' to numpy.int64
    for i in range(0, len(df['service'])):
        if not df['service'][i]:
            df['service'][i] = np.int64(0)
        else:
            if type(df['service'][i]) != np.int64:
                df['service'][i] = np.int64(df['service'][i])

    # Convert 'attack_cat' to numpy.int64
    for i in range(0, len(df['attack_cat'])):
        if not df['attack_cat'][i]:
            df['attack_cat'][i] = np.int64(0)
        else:
            if type(df['attack_cat'][i]) != int:
                df['attack_cat'][i] = np.int64(df['attack_cat'][i])


    # Create label feature
    if 'label' in df.columns.values:
        pass
    else:
        print("Creating 'label' feature in CSV file")
        df['label'] = ""

    # Process label
    for i in range(0, len(df['attack_cat'])):
        if df['attack_cat'][i] == 0 or df['attack_cat'][i] == 9:
            df['label'][i] = np.int64(0)
        else:
            df['label'][i] = np.int64(1)


    # Convert 'label' to numpy.int64
    for i in range(0, len(df['label'])):
        if not df['label'][i]:
            df['label'][i] = np.int64(0)
        else:
            if type(df['label'][i]) != int:
                df['label'][i] = np.int64(df['label'][i])

    
    # Convert 'is_ftp_login' to numpy.int64
    for i in range(0, len(df['is_ftp_login'])):
        if not df['is_ftp_login'][i]:
            df['is_ftp_login'][i] = np.int64(0)
        else:
            if type(df['is_ftp_login'][i]) != int:
                df['is_ftp_login'][i] = np.int64(df['is_ftp_login'][i])


    # Convert 'is_sm_ips_ports' to numpy.int64
    for i in range(0, len(df['is_sm_ips_ports'])):
        if not df['is_sm_ips_ports'][i]:
            df['is_sm_ips_ports'][i] = np.int64(0)
        else:
            if type(df['is_sm_ips_ports'][i]) != int:
                df['is_sm_ips_ports'][i] = np.int64(df['is_sm_ips_ports'][i])

    
    # Convert 'ct_ftp_cmd' to numpy.int64
    for i in range(0, len(df['ct_ftp_cmd'])):
        if not df['ct_ftp_cmd'][i]:
            df['ct_ftp_cmd'][i] = np.int64(0)
        else:
            if type(df['ct_ftp_cmd'][i]) != int:
                df['ct_ftp_cmd'][i] = np.int64(df['ct_ftp_cmd'][i])


    # Drop unnecessary rows (empty or unusable)
    try:
        if "id" in df.columns.values:
            for i in range(0, len(df['id'])):
                if df['proto'][i] == "dropthisrow":
                    df = df.drop(df['id'][i])
    except:
        pass



    # save to csv file
    if 'id' in df.columns.values:
        df.to_csv("./data/mega_detection.csv", index=False)
        print("Changes saved with id=False")
    else:
        df.to_csv("./data/mega_detection.csv", index=True, index_label='id')
        print("Changes saved with id=True, index_label='id'")
        print("Please run script again to drop unnecessary rows")


data_processing()