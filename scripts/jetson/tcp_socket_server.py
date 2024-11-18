import socket
import struct
import pickle
import time

from trunk_width_estimation import TrunkAnalyzer



trunk_segmenter = TrunkAnalyzer(combine_segmenter=True)

server_address = ('', 65432)  # Bind to all interfaces

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(server_address)
sock.listen(1)

times = []
print("------")
print("Waiting for a connection...")
    # locations, widths, classes, img_x_positions, seg_img = trunk_segmenter.pf_helper(depth_image, rgb_image=rgb_image)
    

while True:
    connection, client_address = sock.accept()
    print(f"Connection from {client_address}")
    try:
        while True:
            # Receive the length of the serialized data first
            data_length = struct.unpack("I", connection.recv(4))[0]

            # Receive the serialized data
            data = b''
            while len(data) < data_length:
                packet = connection.recv(data_length - len(data))
                if not packet:
                    break
                data += packet

            if len(data) != data_length:
                print("Incomplete data received, closing connection")
                break

            # Deserialize the received data
            received_data = pickle.loads(data)
            
            timer_start = time.time()
            rgb_image = received_data['rgb_image']
            depth_image = received_data['depth_image']
            
            locations, widths, classes, img_x_positions, seg_img = trunk_segmenter.pf_helper(depth_image, rgb_image=rgb_image)

            # Prepare the response data
            response_data = {
                'positions': locations,
                'widths': widths,
                'class_estimates': classes,
                'img_x_positions': img_x_positions,
                'seg_img': seg_img
            }
            
            time_end = time.time()
            
            serialized_response = pickle.dumps(response_data)
            response_length = len(serialized_response)

            # Send the length of the serialized response first
            connection.sendall(struct.pack("I", response_length))
            # Send the serialized response
            connection.sendall(serialized_response)
            print("Response sent")
            print(f"Time taken: {time_end - timer_start}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()  # Ensure the connection is closed
        print(f"Connection with {client_address} closed")




