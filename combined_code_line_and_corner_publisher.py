"""

"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import qos_profile_sensor_data

class LineAndCOrnerPublisher(Node):
    def __init__(self):
        super().__init__('Line_and_corner_detector')

        # Initialisation
        self.bridge = CvBridge()
        self.depth_image = None
       
        # Parameters
        self.threshold:int = 20     # Minimum size of object (pixels)
        self.step:int = 3           # pixel step for 8-segment
        self.cx:float=0.00
        self.cy:float=0.00
       
        # Subscriptions
        self.rgb_sub = self.create_subscription(Image,'/zed/zed_node/right/image_rect_color',self.rgb_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(Image,'/zed/zed_node/depth/depth_registered',self.depth_callback, qos_profile_sensor_data)
        self.get_logger().info("Subscribed to RGB and Depth topics.")

        # Publishers
        self.is_corner_detected_pub = self.create_publisher(Int32,'/yellow_line/is_corner', 10)
        self.corner_distance_pub = self.create_publisher(Float32,'/yellow_line/corner_distance', 10)

        # ORB, which is an algorithm that detects key, or notable features in controus
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.get_logger().info("Yellow L-corner & Depth node initialized.")

        # Check to see if the centroid is in the yellow region
        self.is_centroid_not_yellow = False
        # Check to see if a corner is detected
        self.is_yellow_corner_detected = False

        # Default depth message
        self.yellow_corner_dist_msg = 100.00


    def depth_callback(self, msg):
        # Get the depth image
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

   
    def rgb_callback(self, msg):
        # frame is the rgb image from zed
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
           
        if frame is None:
            return
       
        if self.depth_image is None:
            self.get_logger().warn("Depth image not yet received")
            return
       

        ### Pre processing steps ###

        # Get HSV image
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # HSV ranges for yellow colour
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])

        # Create a mask, so that only yellow is 255 and rest 0
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        # Finding the largest contour
        if len(contours) != 0:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]

            # Calculate a bounding box around the largest contour
            x, y, w, h = cv2.boundingRect(cnt)
            if ((w > self.threshold or h > self.threshold)):

                # Find the center of the bounding rectangle
                self.cx, self.cy = x + w // 2, y + h // 2

                # Calculate the depth of the center using the depth image
                Z = self.depth_image[self.cy,self.cx]

                # Display the bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                label = f"z:{Z}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2, cv2.LINE_AA)
               
                # Make a mask such that only the largest yellow region is 255
                largest_mask = np.zeros_like(mask)
                cv2.drawContours(largest_mask, [cnt], -1, 255, thickness=cv2.FILLED)

                # Check where the centroid is present
                if largest_mask[int(self.cy), int(self.cx)] == 0:
                    self.is_centroid_not_yellow = True
                else:
                    self.is_centroid_not_yellow = False
                cv2.circle(frame,(int(self.cx), int(self.cy)), 8, (255, 0, 0), 3)
                # Centroid will be (255,0,0) unless overridden

                if (self.is_centroid_not_yellow):

                    # Find the yellow region in frame and keep it yellow
                    yellow_region = cv2.bitwise_and(frame, frame, mask=largest_mask)

                    # Convert to grayscale
                    gray_yellow = cv2.cvtColor(yellow_region, cv2.COLOR_BGR2GRAY)

                    # Apply CLAHE for local contrast enhancement
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

                    # Enhance I suppose
                    enhanced = clahe.apply(gray_yellow)

                    # Gaussian Blur
                    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

                    # Yellow cv2.KeyPoints converted to numpy array
                    yellow_kps = cv2.KeyPoint_convert(self.orb.detect(blurred, mask=largest_mask))  #initially --> keypoints = self.orb.detect(blurred, None)

                    # Prevent index errors
                    if len(yellow_kps) == 0:
                        return

                    best_corner = None
                    corner_list = []

                    ### Main 8-segment code ###

                    # All the directions to check
                    offsets = np.array([
                        (1, 0),
                        (1, 1),
                        (0, 1),
                        (-1, 1),
                        (-1, 0),
                        (-1, -1),
                        (0, -1),
                        (1, -1)], dtype=int) * self.step
                    

                    # find all the eight points
                    centers = np.rint(yellow_kps).astype(np.int32)
                    points = centers[:, None, :] + offsets

                    # Just a small check to see if the points lie within the image bounds
                    h, w = largest_mask.shape
                    points = np.clip(points, [0, 0], [w - 1, h - 1])

                    # Get mask values
                    mask_values = largest_mask[points[..., 1], points[..., 0]]
                   
                    # Convert mask value to binary
                    binary = (mask_values == 255).astype(np.uint8)

                    # Find count for each point in mask
                    counts = binary.sum(axis=1)

                    # Make valid corners list
                    corner_list = centers[(counts <= 4) & (centers[:, 0] <= largest_mask.shape[1] // 2)]
                   
                   # Put circles for debugging
                    for corner in corner_list:
                        cv2.circle(frame, tuple(corner), 6, (255, 0, 0), 5)

                    # Circle for the corner
                    if corner_list.size:

                        # best_corner_tuple=min(corner_list, key=lambda x: abs(x[2] - 90.0))
                        best_corner = tuple(corner_list[0])

                        cv2.circle(frame, best_corner, 8, (255, 0, 255), 1)
                        corner_z = self.depth_image[best_corner[1], best_corner[0]]
                        cv2.putText(frame, f"corner z:{corner_z}", (best_corner[0] + 10, best_corner[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        self.is_yellow_corner_detected = True
                        self.yellow_corner_dist_msg = corner_z
                       
                    # Center point visualisation    
                    cv2.circle(frame,(int(self.cx), int(self.cy)), 8, (255, 0, 255), 3)

                else:
                    self.is_yellow_corner_detected = False
                    self.yellow_corner_dist_msg = 100.00
               
        # Images to visualise what is happening
        cv2.imshow("frame", frame)
        # cv2.imshow("mask2", mask2)
        cv2.imshow("mask", mask)
        cv2.waitKey(1)

        # Messages to published
        msg1 = Bool()
        msg1.data = self.is_yellow_corner_detected
        msg2 = Float32()
        msg2.data = float(self.yellow_corner_dist_msg)
        self.is_corner_detected_pub.publish(msg1)
        self.corner_distance_pub.publish(msg2)



def main(args=None):
    rclpy.init(args=args)
    node = LineAndCOrnerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()