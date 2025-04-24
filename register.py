import os
import cv2
from hardware import OV5647Controller
from face_detector import FaceDetector
from face_recognitiona import FaceRecognizer

def main(hardware=None):
    if hardware is None:
        hardware = OV5647Controller()
    #hardware = OV5647Controller()
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    save_dir = "/home/pi/Desktop/python/lian5huoti/faces"
    os.makedirs(save_dir, exist_ok=True)
    
    
    user_id = input("qingshuru yonghu id: ")
    
    if user_id.lower() == "reset":
        if os.path.exists("embeddings.pkl"):
            os.remove("embeddings.pkl")
            print("yi jing qingchu shuoyou zucheshu ju")
        return
    
    cv2.namedWindow("Registration Preview", cv2.WINDOW_NORMAL)
    
    capture_frame = None
    while True:
        frame = hardware.capture_frame()
        if frame is None:
            continue
        display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        faces = detector.detect_faces(display_frame, return_coords=True)
        if faces:
            (x1, y1, x2, y2) = faces[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        else:
            cv2.putText(frame, "No Face Detected", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
#         faces = detector.detect_faces(display_frame, return_coords=True)
#         if faces:
#             (x1, y1, x2, y2) = faces[0]
#             cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(display_frame, "Face Detected", (x1, y1-10),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
#         else:
#             cv2.putText(display_frame, "No Face Detected", (20, 40),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        
#         cv2.imshow("Registration Preview", display_frame)
        cv2.imshow("Registration Preview", frame)
        key = cv2.waitKey(1)
        if key == ord(' '):
            if faces:
                capture_frame = frame.copy()
                break
            else:
                print("weijiance daolian,qing tiaozheng")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return
    if len(capture_frame.shape) == 2:
        capture_frame = cv2.cvtColor(capture_frame, cv2.COLOR_GRAY2BGR)
    
    faces = detector.detect_faces(capture_frame, return_coords=True)
    if not faces:
        print("weijiance daolian")
        return
    x1, y1, x2, y2 = faces[0]
    face_roi = capture_frame[y1:y2, x1:x2]
    save_path = os.path.join(save_dir, f"{user_id}.jpg")
    cv2.imwrite(save_path, frame)
    #cv2.imwrite(save_path, cv2.cvtColor(capture_frame, cv2.COLOR_RGB2BGR))
    print(f"renliantu pian chunzhi: {save_path}")
    
    if recognizer.register(user_id, face_roi):
        emb = recognizer.embeddings[user_id]
        if len(emb) == 192:
            print(f"yonghu {user_id} zucheng chenggong (192weitezheng)")
        else:
            print(f"zuce shibai: tezheng weidu yichang {len(emb)}wei")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
        
#     frame = hardware.capture_frame()
#     if frame is None:
#         print("tupian puhu shibai")
#         return
#     
#     
#     if len(frame.shape) == 2:
#         frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#     
#     #faces = detector.detect_faces(frame)
#     faces = detector.detect_faces(frame, return_coords=True) 
#     
#     if not faces:
#         print("weijiance daolian")
#         return
#     
# 
#     
#     x1, y1, x2, y2 = faces[0]
#     face_roi = frame[y1:y2, x1:x2]
#     
#     display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
# #     cv2.imshow("zuce renlian",display_frame)
# #     cv2.waitKey(3000)
#     save_path = os.path.join(save_dir, f"{user_id}.jpg")
#     cv2.imwrite(save_path, display_frame)
#     print(f"renliantu pian chunzhi: {save_path}")
#     
#     if recognizer.register(user_id, face_roi):
#         emb = recognizer.embeddings[user_id]
#         if len(emb) == 192:
#             cv2.destroyAllWindows()
#             print(f"yonghu {user_id} zucheng chenggong (192weitezheng)")
#         else:
#             print("zuce shibai: tezheng weidu yichang{len(emb)}wei")
#         #del recognizer.embeddings[user_id]
#         
# if __name__ == "__main__":
#     main()
#     
    
