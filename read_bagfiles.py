import nml_bag
import os
HOME  = os.path.abspath(os.path.dirname(__file__))
file = os.path.join(HOME,'Aufnahmen/rosbag2_2023_06_29-13_53_05/rosbag2_2023_06_29-13_53_05_0.db3')

reader = nml_bag.Reader('path/to/bag.db3', topics=['/vesc/odom'])
for message_record in reader: print(message_record)