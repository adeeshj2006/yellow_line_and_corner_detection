import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class YellowDepthDetector(Node):
    def __init__(self):
        super().__init__('yellow_depth_detector')
        self.bridge = CvBridge()
        self.depth_image = None
        self.threshold = 20  # Minimum size of object (pixels)
        self.cx=0.00
        self.cy=0.00
        self.check = False
        self.rgb_sub = self.create_subscription(Image,'/zed/zed_node/right/image_rect_color',self.rgb_callback,10)
        self.depth_sub = self.create_subscription(Image,'/zed/zed_node/depth/depth_registered',self.depth_callback,10)
        self.get_logger().info("Subscribed to RGB and Depth topics.")
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.get_logger().info("Yellow L-corner & Depth node initialized.")
        self.step = 3
        

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')


    # def angle_between(self, p1, p2, p3):
    #     a = np.array(p1) - np.array(p2)
    #     b = np.array(p3) - np.array(p2)
    #     dot = np.dot(a, b)
    #     norm = np.linalg.norm(a) * np.linalg.norm(b)
    #     if norm == 0:
    #         return 0
    #     cos_angle = np.clip(dot / norm, -1.0, 1.0)
    #     angle = np.degrees(np.arccos(cos_angle))
    #     return min(angle, 360 - angle)
    
    
    def rgb_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
        if frame is None:
            return
    
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = np.ones((5, 5), np.uint8)
        mask2 = cv2.dilate(mask, kernel, iterations=2)
        mask2= cv2.erode(mask2, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if self.depth_image is None:
            self.get_logger().warn("Depth image not yet received")
            return
        
        if len(contours)!=0:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt=contours[max_index]
            x, y, w, h = cv2.boundingRect(cnt)
            if ((w > self.threshold or h > self.threshold)):
                self.cx, self.cy = x + w // 2, y + h // 2
                Z=self.depth_image[self.cy,self.cx]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                label = f"z:{Z}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2, cv2.LINE_AA)

                largest_mask = np.zeros_like(mask)
                cv2.drawContours(largest_mask, [cnt], -1, 255, thickness=cv2.FILLED)
                # bool for corner detection --> self.check
                if mask[int(self.cy), int(self.cx)] == 0:
                    self.check = True
                else:
                    self.check = False

                if (self.check):
                    yellow_region = cv2.bitwise_and(frame, frame, mask=largest_mask)
                    gray_yellow = cv2.cvtColor(yellow_region, cv2.COLOR_BGR2GRAY)
                    # Apply CLAHE for local contrast enhancement
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(gray_yellow)
                    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
                    yellow_kps = self.orb.detect(blurred, mask=largest_mask)  #initially --> keypoints = self.orb.detect(blurred, None)
                    
                    best_corner = None
                    corner_list = []

                    # Here is the one using 8 segments
                    count = 0
                    for kp in yellow_kps:
                        # Point coords
                        xk, yk = int(kp.pt[0]), int(kp.pt[1])

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
                            count += mask[int(step[1]), int(step[0])]

                        if count <= 4:
                            corner_list.append(kp)

                        count = 0
                    
                    print("Corners: ", len(corner_list))
                    
                    for corner in corner_list:
                        cv2.circle(frame, (int(corner.pt[0]), int(corner.pt[1])), 15, (255, 0, 0), 5)

                    if corner_list:
                        # best_corner_tuple=min(corner_list, key=lambda x: abs(x[2] - 90.0))
                        best_corner=(int(corner_list[0].pt[0]), int(corner_list[0].pt[1])) ## This was the error
                        cv2.circle(frame, best_corner, 8, (255, 0, 255), 1)
                        corner_z=self.depth_image[best_corner[1],best_corner[0]]
                        cv2.putText(frame, f"corner z:{corner_z}", (best_corner[0] + 10, best_corner[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        
                    # Center point visualisation    
                    cv2.circle(frame,(int(self.cx), int(self.cy)), 8, (255, 0, 255), 3)
        
        cv2.imshow("frame", frame)
        cv2.imshow("mask2", mask2)
        cv2.imshow("mask", mask)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YellowDepthDetector()
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

