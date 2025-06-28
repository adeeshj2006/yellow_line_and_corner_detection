 def yolo(source)
        yolo = YOLO("path of yolov8n.engine")
        self.get_logger().info("yolo engine loaded")

        '''
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #cv_img = cv2.resize(cv_img, (640, 640))
        source = cv2.imread(cv_img)
        '''

        results = yolo.predict(source, imgsz=640, conf=0.2)
        img = results[0].plot()
        cv2.imshow("pred", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = box[:4]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            print(f"Center: ({cx.item():.1f}, {cy.item():.1f})")
