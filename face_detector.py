import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",  
            "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        )
        #self.conf_threshold = 0.50
        self.conf_threshold = 0.30
        
        self.min_face_size = 100

    def detect_faces(self, frame,return_coords=True):
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#         if frame.dtype != np.uint8:
#             frame = frame.astype(np.uint8)
            
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0),
            swapRB=True )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        valid_faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                    continue
                
#                 if return_coords:
#                     valid_faces.append((x1, y1, x2, y2))
#                 else:
#                     face_roi = frame[y1:y2, x1:x2]
#                     valid_faces.append(face_roi)
                    
                valid_faces.append((x1, y1, x2, y2))
        
#         return valid_faces
        if return_coords:
            return valid_faces
        else:
            return [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in valid_faces]
            
                
                