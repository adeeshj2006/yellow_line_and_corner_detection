import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math
import numpy as np

# Upper and Lower limits of yellow. Tune these:
lower_yellow = np.array([20,100,100])
upper_yellow = np.array([34,255,255])

class YellowLineDepthDetector(Node):
    def __init__(self):
        super().__init__('yellow_line_depth_detector')
        
        # Initialising some variables
        self.bridge = CvBridge()
        self.depth_image = None
        self.output = True                # Do you want CV2 imshow output
        self.debug = False
        
        # Parameters
        self.pixel_threshold = 50           # Pixel threshold for contour. Won't form contour if both height and width is less than this value
        self.bounding_box_ratio = 5.0       # Won't form bounding box if width/height ratio is less than this
        self.step = 3 
        self.cx=0.00
        self.cy=0.00
        self.is_centroid_not_yellow = False
        self.is_yellow_line_detected: bool = False
        self.yellow_line_dist: float = 100.0
        self.is_yellow_corner_detected = False
        self.yellow_corner_dist_msg = 100.00

        self.orb = cv2.ORB_create(nfeatures=1000)
        # # Camera focal length and principal coordinate positions (both x and y)
        # self.fx = 130.5836
        # self.fy = 130.5837
        # self.cx = 167.7724
        # self.cy = 93.2384

        # Subscribers
        self.rgb_image_sub = self.create_subscription(
            Image,
            '/zed/zed_node/right/image_rect_color',
            self.rgb_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/zed/zed_node/depth/depth_registered',
            self.depth_callback,
            10
        )

        # Publishers
        self.is_yellow_line_detected_pub = self.create_publisher(
            Int32,
            '/yellow_line/is_yellow_line',
            10
        )
        self.yellow_line_distance_pub = self.create_publisher(
            Float32,
            '/yellow_line/yellow_line_distance',
            10
        )
        self.is_corner_detected_pub = self.create_publisher(
            Int32,
            '/yellow_line/is_corner',
            10
        )
        self.corner_distance_pub = self.create_publisher(
            Float32,
            '/yellow_line/corner_distance',
            10
        )
        self.is_corner_detected_pub = self.create_publisher(
            Int32,
            '/yellow_line/is_corner',
            10
        )
        self.corner_distance_pub = self.create_publisher(
            Float32,
            '/yellow_line/corner_distance',
            10
        )
        
        self.get_logger().info('Initialisation Complete')

    def depth_callback(self, msg:Image) -> None:
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth Image could not be converted: {e}')



    def rgb_callback(self,msg:Image) -> None:
        frame = None

        # Convert ROS Image to OpenCV Format (frame)
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Frame could not be converted: {e}')

        if frame is None:
            self.get_logger().error('Frame was None after conversion')
            return

        # Image preprocessing and conversion
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If depth image has not been received -> Raise error
        if self.depth_image is None:
            self.get_logger().error('Depth Image not received')
            return
        all_possible_contours = []

        if len(contours) != 0:          # There are non-zero parent contours
            if (self.is_yellow_line_detected == True):
                self.bounding_box_ratio = -1
            for c in contours:
                x,y,w,h = cv2.boundingRect(c)
                if 1.0*w/h > self.bounding_box_ratio:
                    all_possible_contours.append(c)
            if len(all_possible_contours) !=0:
                areas = [cv2.contourArea(c) for c in all_possible_contours]
                max_index = np.argmax(areas)

            else:
                areas = []
                max_index = 0
            
            largest_contour = contours[max_index]

            x, y, w, h = cv2.boundingRect(largest_contour)

            if ((w > self.pixel_threshold or h > self.pixel_threshold)):
                self.get_logger().info("Perfect Condition")
                self.cx, self.cy = x + w // 2, y + h // 2

                self.is_yellow_line_detected = True
                self.yellow_line_dist = float(self.depth_image[self.cy,self.cx])

                if (math.isnan(self.yellow_line_dist) or math.isinf(self.yellow_line_dist)):

                    self.get_logger().error(f'Skipping distance value: {self.yellow_line_dist}')
                    return

                # Drawing the bounding box
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)

                # Depth label
                label = f'Distance:{self.yellow_line_dist}'
                # print("Distance: ",self.yellow_line_dist)
                cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            
                #majority of corner detection start here
                
                largest_mask = np.zeros_like(yellow_mask)
                cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

                if yellow_mask[int(self.cy), int(self.cx)] == 0:
                    self.is_centroid_not_yellow = True
                else:
                    self.is_centroid_not_yellow = False

                cv2.circle(frame,(int(self.cx), int(self.cy)), 8, (255, 0, 0), 3)           # Centroid will be (255,0,0) unless overridden

                if (self.is_centroid_not_yellow):
                    yellow_region = cv2.bitwise_and(frame, frame, mask=largest_mask)
                    gray_yellow = cv2.cvtColor(yellow_region, cv2.COLOR_BGR2GRAY)

                    # Apply CLAHE for local contrast enhancement
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(gray_yellow)
                    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

                    yellow_kps = self.orb.detect(blurred, mask=largest_mask)  #initially --> keypoints = self.orb.detect(blurred, None)
                    
                    best_corner = None
                    corner_list = []

                    # Main 8-segment code
                    count = 0
                    
                    i=0
                    for kp in yellow_kps:
                        i += 1
                        # Point coords
                        xk, yk = int(kp.pt[0]), int(kp.pt[1])
                        if self.debug:
                            cv2.circle(frame, (xk, yk), 2, (100,100,100), 2)
                            cv2.rectangle(frame, (xk + self.step, yk + self.step), (xk - self.step, yk - self.step), (0,0,255), 2)
                            cv2.putText(frame, f'kp_{i}', (xk, yk), 1, 1, (0,0,0), 1)

                        steps = [
                            (xk + self.step, yk),
                            (xk + self.step, yk + self.step),
                            (xk, yk + self.step),
                            (xk - self.step, yk + self.step),
                            (xk - self.step, yk),
                            (xk - self.step, yk - self.step),
                            (xk, yk - self.step),
                            (xk + self.step, yk - self.step)
                        ]

                        for step in steps:
                            count += int(yellow_mask[int(step[1]), int(step[0])] / 255)          # Counts the number of segments which are within the mask
                        
                        print (f'Count for kp_{i}: {count}')
                        if count <= 4:
                            corner_list.append(kp)

                        count = 0
                    
                    # print("Corners: ", len(corner_list))
                    if self.debug:
                        for corner in corner_list:
                            cv2.circle(frame, (int(corner.pt[0]), int(corner.pt[1])), 15, (255, 0, 0), 5)

                    if corner_list:
                        # best_corner_tuple=min(corner_list, key=lambda x: abs(x[2] - 90.0))
                        best_corner=(int(corner_list[0].pt[0]), int(corner_list[0].pt[1])) ## This was the error
                        cv2.circle(frame, best_corner, 8, (255, 0, 255), 1)
                        corner_z=self.depth_image[best_corner[1],best_corner[0]]
                        cv2.putText(frame, f"corner z:{corner_z}", (best_corner[0] + 10, best_corner[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        self.is_yellow_corner_detected = True 
                        self.yellow_corner_dist_msg = corner_z
                        # pass
                    # Center point visualisation    
                    cv2.circle(frame,(int(self.cx), int(self.cy)), 8, (255, 0, 255), 3)
                

            
            else:
                # Contour too small
                # print('Line not detected. Z is 100')
                self.is_yellow_corner_detected = False
                self.yellow_corner_dist_msg = 100.00
                self.yellow_line_dist = 101.0
                self.is_yellow_line_detected = False
                self.get_logger().warn(f'Line not detected')

            # CV2 Output
            if self.output:
                cv2.imshow("Camera feed", frame)
                cv2.imshow("Yellow mask", yellow_mask)
                cv2.waitKey(1)


        # Publishing
        msg1 = Int32()
        if self.is_yellow_corner_detected:
            msg1.data = 1
        else:
            msg1.data = 0
        msg2 = Float32()
        msg2.data = float(self.yellow_corner_dist_msg)
        self.is_corner_detected_pub.publish(msg1)
        self.corner_distance_pub.publish(msg2)

        is_yellow_line_detected_msg = Int32()
        if self.is_yellow_line_detected:
            is_yellow_line_detected_msg.data = 1
        else:
            is_yellow_line_detected_msg.data = 0
        yellow_line_dist_msg = Float32()
        yellow_line_dist_msg.data = self.yellow_line_dist
        self.is_yellow_line_detected_pub.publish(is_yellow_line_detected_msg)
        self.yellow_line_distance_pub.publish(yellow_line_dist_msg)
        self.get_logger().info(f'Dist: {self.yellow_line_dist}')

def main(args=None):
    rclpy.init(args=args)
    node = YellowLineDepthDetector()

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
