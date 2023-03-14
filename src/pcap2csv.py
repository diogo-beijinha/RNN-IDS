import os
import pathlib
import subprocess
from termcolor import colored

import pandas as pd
import numpy as np

# Change this to your Tshark Path
tshark_path = "C:\Program Files\Wireshark\tshark.exe"
print("TShark path set to:",tshark_path)

# Change this to your PCAP folder path
pcap_path = "D:\Programming Projects\Python\RNN-IDS-Model\src\data\pcap"
print(colored("PCAP folder path set to:",'blue', attrs=['underline','bold']), colored(pcap_path, 'yellow'))

# Change this to your CSV folder path
csv_path = "D:\Programming Projects\Python\RNN-IDS-Model\src\data"
print(colored("CSV folder path set to:",'blue', attrs=['underline','bold']), colored(csv_path, 'yellow'))
pcap_files_array = []


if os.path.exists(pcap_path):
    print(colored("PCAP FOLDER FOUND",'blue', attrs=['bold']))
    files_array = dir_list = os.listdir(pcap_path)
    # Add pcap files from folder to array
    for file in files_array:
        file_extension = pathlib.Path(file).suffix
        if file_extension == ".pcap" or file_extension == ".pcapng" or file_extension == ".dmp":
            pcap_files_array.append(file)
    print(colored("Found", 'blue', attrs=['bold']), colored(len(pcap_files_array), 'yellow', attrs=['bold']) ,colored("PCAP FILES", 'blue', attrs=['bold']))
    
    # Using TShark get all the necessary data from the PCAP files and save them in the CSV file
    for file in pcap_files_array:
        file_name = os.path.splitext(file)[0]
        csv_file_path = csv_path + "\\" + file_name + ".csv"
        print(colored("Getting Features for:", 'blue', attrs=['bold', 'underline']), colored(file_name + '.csv', 'yellow'))
        subprocess.call("C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe cd 'C:\Program Files\Wireshark\'; ./tshark.exe -r '{}\{}' -T fields -E header=y -E separator=',' -E quote=n -E occurrence=f -e frame.time_delta -e ip.ttl -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e tcp.connection.syn -e ip.len -e ip.flags.df -e ip.flags.mf -e ip.fragment -e frame.len -e ip.fragment.count -e ip.fragments -e frame.protocols -e frame.time_relative -e tcp.window_size -e tcp.ack -e tcp.seq -e tcp.len -e tcp.stream -e tcp.urgent_pointer -e tcp.flags -e tcp.analysis.ack_rtt -e tcp.segments -e tcp.reassembled.length -e http.request -e tcp.port -e udp.port -e arp.dst.hw_mac -e icmp.type -e icmp.code -e tcp.flags.syn -e tcp.flags.ack -e tcp.flags.fin -e tcp.flags.push -e tcp.flags.urg -e arp.duplicate-address-detected -e arp.duplicate-address-frame -e tcp.analysis.ack_lost_segment -e tcp.analysis.retransmission -e tcp.analysis.fast_retransmission -e tcp.analysis.spurious_retransmission -e icmp -e data.len -e dtp -e vlan.too_many_tags -e tcp.analysis.lost_segment -e tcp.analysis.retransmission -e wlan.fc.type_subtype -e tcp.connection.fin -e tcp.connection.rst -e ftp.response.code -e arp.opcode -e tcp.flags.reset -e http.request.method -e ftp.command-frame -e dns.resp.ttl > '{}'".format(pcap_path, file,csv_file_path), shell=False)

    mega_df = pd.DataFrame()

    for file in pcap_files_array:
        file_name = os.path.splitext(file)[0]
        csv_file_path = csv_path + "\\" + file_name + ".csv"
        df = pd.read_csv(csv_file_path, encoding="UTF-16", keep_default_na=False, na_values='')
        if len(df) > 3000:
            mega_df = mega_df.append(df[0:2999], ignore_index=True)
        else:
            mega_df = mega_df.append(df, ignore_index=True)
    
    # Save data to CSV file
    mega_df.to_csv("./data/mega_detection.csv", index=False)
    print(colored("CHANGES SAVED WITH",'blue', attrs=['bold']),colored("INDEX=",'green', attrs=['bold']),colored("FALSE",'yellow', attrs=['bold']))
    
    # Delete temporary CSV files created
    for file in pcap_files_array:
        file_name = os.path.splitext(file)[0]
        csv_file_path = csv_path + "\\" + file_name + ".csv"
        os.remove(csv_file_path)
    

else:
    print("NO PCAP FOLDER FOUND!")

