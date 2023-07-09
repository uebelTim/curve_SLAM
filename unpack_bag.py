# import nml_bag

# reader = nml_bag.Reader('/home/ue-ubuntu/Documents/Aufnahmen/rosbag2_2023_06_29-13_55_44',
#                          topics=['/vesc/ackermann_cmd', '/debug_image','/camera/imu','/vesc/odom','/camera/color/image_raw'])

# for i,message_record in enumerate(reader):
#     print(message_record)
#     if i>2:
#         break

import rosbag2_py
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
import cv_bridge
import numpy as np
import os
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from PIL import Image
import pandas as pd

bridge = cv_bridge.CvBridge()

# Initialize ROS
rclpy.init()

# Create a reader
reader = rosbag2_py.SequentialReader()

# Open the bag file
storage_options = rosbag2_py.StorageOptions(uri="/home/ue-ubuntu/Documents/Aufnahmen/rosbag2_2023_06_29-13_55_44", storage_id="sqlite3")
converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
reader.open(storage_options, converter_options)

i=0
raw = 0
debug = 0
if os.path.exists('/home/ue-ubuntu/Desktop/recordings/raw'):
    os.system('rm -r /home/ue-ubuntu/Desktop/recordings/raw')
os.makedirs('/home/ue-ubuntu/Desktop/recordings/raw', exist_ok=True)
if os.path.exists('/home/ue-ubuntu/Desktop/recordings/debug'):
    os.system('rm -r /home/ue-ubuntu/Desktop/recordings/debug')
os.makedirs('/home/ue-ubuntu/Desktop/recordings/debug', exist_ok=True)
# odom = pd.DataFrame(columns=['x','y','z','qx','qy','qz','qw'])
# ackermann = pd.DataFrame(columns=['speed','steering_angle'])
# imu = pd.DataFrame(columns=['angular_velocity_x','angular_velocity_y','angular_velocity_z','linear_acceleration_x','linear_acceleration_y','linear_acceleration_z'])
odom = []
ackermann = []
imu = []
while reader.has_next():
    topic, msg, t = reader.read_next()
    print('topic:', topic)
    # if '/camera/color/image_raw' in topic:
    #     Deserialize the message
    #     msg_type = get_message('sensor_msgs/msg/Image')
    #     image_msg = deserialize_message(msg, msg_type)
    #     Manually set the encoding
    #     deserialized_msg.encoding = 'bgr8'

    #     Convert the ROS image message to an OpenCV image
    #     cv_image = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
    #     plt.imshow(cv_image)
    #     plt.show()

    #     Save image to ~/Desktop/recordings/raw make sure the folder exists, create it if not
    #     delete the folder if it exists
    #     plt.imsave('/home/ue-ubuntu/Desktop/recordings/raw/image_'+str(raw)+'.png', cv_image)
    #     Image.fromarray(cv_image).save('/home/ue-ubuntu/Desktop/recordings/raw/image_'+str(raw)+'.png')
    #     raw+=1
    # if '/debug_image' in topic:
    #     msg_type = get_message('sensor_msgs/msg/Image')
    #     image_msg = deserialize_message(msg, msg_type)
    #     cv_image = bridge.imgmsg_to_cv2(image_msg, '8UC1')
    #     plt.imshow(cv_image, cmap='gray')
    #     plt.show()
    #     plt.imsave('/home/ue-ubuntu/Desktop/recordings/debug/image_'+str(raw)+'.png', cv_image)
    #     Image.fromarray(cv_image).save('/home/ue-ubuntu/Desktop/recordings/debug/image_'+str(raw)+'.png')
    #     debug+=1

    if '/vesc/odom' in topic:
        msg_type = get_message('nav_msgs/msg/Odometry')
        msg = deserialize_message(msg, msg_type)
        print('odom pose:', msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z)
        print('odom orientation:', msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        odom.append([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        
    if '/vesc/ackermann_cmd' in topic:
        msg_type = get_message('ackermann_msgs/msg/AckermannDriveStamped')
        msg = deserialize_message(msg, msg_type)
        print('ackermann speed', msg.drive.speed)
        print('ackermann steering angle', msg.drive.steering_angle)
        ackermann.append([msg.drive.speed, msg.drive.steering_angle])

    if '/camera/imu' in topic:
        msg_type = get_message('sensor_msgs/msg/Imu')
        msg = deserialize_message(msg, msg_type)
        print('imu angular_velocity', msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)
        print('imu linear_acceleration', msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z)
        imu.append([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z, msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        
    # i+=1
    # if i>30:
    #     break

#odom to dataframe
odom = pd.DataFrame(odom, columns=['x','y','z','qx','qy','qz','qw'])
ackermann = pd.DataFrame(ackermann, columns=['speed','steering_angle'])
imu = pd.DataFrame(imu, columns=['angular_velocity_x','angular_velocity_y','angular_velocity_z','linear_acceleration_x','linear_acceleration_y','linear_acceleration_z'])
#save dataframes
odom.to_csv('/home/ue-ubuntu/Desktop/recordings/odom.csv', index=False)
ackermann.to_csv('/home/ue-ubuntu/Desktop/recordings/ackermann.csv', index=False)
imu.to_csv('/home/ue-ubuntu/Desktop/recordings/imu.csv', index=False)

# Shutdown ROS
rclpy.shutdown()