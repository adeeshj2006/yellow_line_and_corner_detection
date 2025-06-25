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
        self.check = False;
        self.rgb_sub = self.create_subscription(Image,'/zed/zed_node/right/image_rect_color',self.rgb_callback,10)
        self.depth_sub = self.create_subscription(Image,'/zed/zed_node/depth/depth_registered',self.depth_callback,10)
        self.get_logger().info("Subscribed to RGB and Depth topics.")
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.get_logger().info("Yellow L-corner & Depth node initialized.")
    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def angle_between(self, p1, p2, p3):
        a = np.array(p1) - np.array(p2)
        b = np.array(p3) - np.array(p2)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0
        cos_angle = np.clip(dot / norm, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return min(angle, 360 - angle)
    
    def rgb_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB image conversion failed: {e}")
            

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
        if ((w > self.threshold or h > self.threshold)):# and ((1.0*w)/h)>2):
        #  if ((1.0*w)/h)<2 :
        #     print ("its detecting contour but more like a square")
            self.cx, self.cy = x + w // 2, y + h // 2
            Z=self.depth_image[self.cy,self.cx]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            label = f"z:{Z}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
            if (Z>=3.0):
                pass
                # print("z:",Z)
            else:
                pass
                # print("stop")
            largest_mask = np.zeros_like(mask)
            cv2.drawContours(largest_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            if mask[int(self.cy), int(self.cx)] == 0:
                self.check = True
            else:
                self.check = False
            # print(self.check)
            if (self.check):
                yellow_region = cv2.bitwise_and(frame, frame, mask=largest_mask)
                gray_yellow = cv2.cvtColor(yellow_region, cv2.COLOR_BGR2GRAY)
                # Apply CLAHE for local contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray_yellow)
                blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
                keypoints = self.orb.detect(blurred, None)
                yellow_kps = [kp for kp in keypoints if largest_mask[int(kp.pt[1]), int(kp.pt[0])] > 0]
                for kp in yellow_kps:
                    xk, yk = int(kp.pt[0]), int(kp.pt[1])
                    cv2.rectangle(frame, (xk - 3, yk - 3), (xk + 3, yk + 3), (0, 255, 0), 1)
                # Detect best 90-degree corner from triplets
                best_corner = None
                for i in range(len(yellow_kps)):
                    for j in range(len(yellow_kps)):
                        for k in range(len(yellow_kps)):
                            if i == j or j == k or i == k:
                                continue
                            pt1 = yellow_kps[i].pt
                            pt2 = yellow_kps[j].pt
                            pt3 = yellow_kps[k].pt
                            angle = self.angle_between(pt1, pt2, pt3)
                            if 85 <= angle <= 100:
                                best_corner = (int(pt2[0]), int(pt2[1]))
                                break
                        if best_corner:
                            break
                    if best_corner:
                        break
                if best_corner:
                    cv2.circle(frame, best_corner, 8, (255, 0, 255), 3)
                    cv2.putText(frame, "L-corner", (best_corner[0] + 10, best_corner[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                if best_corner is not None:
                    corner_z=self.depth_image[best_corner[1],best_corner[0]]
                else:
                    corner_z=0
                label = f"z: {corner_z}"
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

