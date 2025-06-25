import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
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
        self.output = True                  # Do you want CV2 imshow output
        
        # Parameters
        self.pixel_threshold = 50           # Pixel threshold for contour. Won't form contour if both height and width is less than this value
        self.bounding_box_ratio = 5.0       # Won't form bounding box if width/height ratio is less than this

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
            Bool,
            '/yellow_line/is_yellow_line',
            10
        )
        self.yellow_line_distance_pub = self.create_publisher(
            Float32,
            '/yellow_line/yellow_line_distance',
            10
        )
        self.is_corner_detected_pub = self.create_publisher(
            Bool,
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
        is_yellow_line_detected: bool = False
        yellow_line_dist: float = 100.0

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
        
        if len(contours) != 0:          # There are non-zero parent contours
            areas = [cv2.contourArea(c) for c in contours]

            max_index = np.argmax(areas)

            largest_contour = contours[max_index]

            x, y, w, h = cv2.boundingRect(largest_contour)

            if ((w > self.pixel_threshold or h > self.pixel_threshold) and 1.0*w/h > self.bounding_box_ratio):
                self.get_logger().info("Perfect Condition")
                center_x, center_y = x + w // 2, y + h // 2

                is_yellow_line_detected = True
                yellow_line_dist = self.depth_image[center_y,center_x]

                if (math.isnan(yellow_line_dist) or math.isinf(yellow_line_dist)):

                    self.get_logger().error(f'Skipping distance value: {yellow_line_dist}')
                    return

                # Drawing the bounding box
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)

                # Depth label
                label = f'Distance:{yellow_line_dist}'
                # print("Distance: ",yellow_line_dist)
                cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
            else:
                # print('Line not detected. Z is 100')
                yellow_line_dist = 101.0
                is_yellow_line_detected = False
                self.get_logger().warn(f'Line not detected')

            # CV2 Output
            if self.output:
                cv2.imshow("Camera feed", frame)
                cv2.imshow("Yellow mask", yellow_mask)
                cv2.waitKey(1)
    
        # Publishing Yellow Line Distnace
        yellow_line_dist_msg = Float32()
        yellow_line_dist_msg.data = yellow_line_dist
        self.is_yellow_line_detected_pub.publish(is_yellow_line_detected)
        self.yellow_line_distance_pub.publish(yellow_line_dist_msg)
        self.get_logger().info(f'Dist: {yellow_line_dist}')

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
