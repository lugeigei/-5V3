import os
import pickle
import logging
import numpy as np
from face_detector import FaceDetector
from sklearn.metrics.pairwise import cosine_similarity

import cv2
import tflite_runtime.interpreter as tflite

class FaceRecognizer:
    def __init__(self):
        self.embeddings = {}
        #self.threshold = 0.75
        self.threshold = 0.58
        self.embedding_dim = 192
        
        self.liveness_threshold = 0.15
        self.detector = FaceDetector()
        
       
        self.interpreter = tflite.Interpreter(model_path="mobilefacenet.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
      
        
        self._load_embeddings()


    def _preprocess_face(self, face_roi):
        resized = cv2.resize(face_roi, (112, 112))
        normalized = (resized.astype(np.float32) - 127.5) / 127.5
        return np.expand_dims(normalized, axis=0)

    def check_liveness(self, frame_sequence):
        if len(frame_sequence) < 3:
            logging.warning("huotijianc yao3zhen")
            return False
        
        embeddings = []
        valid_frames = 0
        for frame in frame_sequence[-3:]:
            faces = self.detector.detect_faces(frame, return_coords=True)
            #faces = self.detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), return_coords=True)
        
            if not faces:
                logging.debug("huoti jiance jieduan  weijiance daolian")
                continue
            
            try:
                x1, y1, x2, y2 = max(faces, key=lambda rect: (rect[2]-rect[0])*(rect[3]-rect[1]))
                if (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue
                
                face_roi = frame[y1:y2, x1:x2]
                face_roi = cv2.resize(face_roi, (112, 112))
                
                emb = self._get_embedding(face_roi)
                if emb is not None:
                    embeddings.append(emb)
            except Exception as e:
                logging.error(f"Face processing error: {str(e)}")
                
        if len(embeddings) < 2:
            return False
        
        diffs = [1 - cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0] 
            for i in range(1, len(embeddings))]
        avg_diff = np.mean(diffs)
        avg_brightness = np.mean(frame_sequence[-1]) / 255
        brightness_factor = np.clip(2.0 - avg_brightness, 0.5, 3.0)
        adjusted_threshold = self.liveness_threshold * brightness_factor
        logging.debug(f"huo tijiance chayi: {avg_diff:.3f}, yuzhi: {adjusted_threshold:.3f}")  
        return avg_diff > adjusted_threshold
    
#                 face_roi = frame[y1:y2, x1:x2]
#                 if face_roi.size == 0:
#                     continue
#                 
#                 face_roi = cv2.resize(face_roi, (112, 112))
#                 #face_roi = cv2.GaussianBlur(face_roi, (5,5), 0)
#                 #face_roi = cv2.medianBlur(face_roi, 3)
# 
#             
#                 emb = self._get_embedding(face_roi)
#                 if emb is not None:
#                     embeddings.append(emb)
#                     valid_frames += 1
#             except Exception as e:
#                 logging.error(f"youxiao zhenchuli yichang: {str(e)}")
#                     
#                 
#         if valid_frames < 2:
#             logging.warning("youxiaozhen buzhu")
#             return False
#         
#         diffs = []
#         for i in range(1, len(embeddings)):
#             try:
#                 similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
#                 diffs.append(1 - similarity)
#             except Exception as e:
#                 logging.error(f"xaingshidujisuan yichang: {str(e)}")
#                 
#         if not diffs:
#             return False
#             
#         avg_diff = np.mean(diffs)
#         avg_brightness = np.mean(frame_sequence[-1]) / 255
#         brightness_factor = np.clip(1.5 - avg_brightness, 0.5, 2.0)
#         adjusted_threshold = self.liveness_threshold * brightness_factor
#         logging.debug(f"huo tijiance pingjjunbianhualv: {avg_diff:.3f}, dongtaiyuzhi: {adjusted_threshold:.3f}")       
#         return avg_diff > adjusted_threshold
    
    def _load_embeddings(self):
        if os.path.exists("embeddings.pkl"):
            with open("embeddings.pkl", "rb") as f:
                self.embeddings = pickle.load(f)
            logging.info(f"yi jiazai {len(self.embeddings)} ge zhuche yonghu")

    def _save_embeddings(self):
        with open("embeddings.pkl", "wb") as f:
            pickle.dump(self.embeddings, f)

    def register(self, user_id, face_roi):
        try:
            embedding = self._get_embedding(face_roi)
            if embedding is not None:
                if user_id in self.embeddings:
                    logging.warning(f"yong hu {user_id} yichun zai,jiangbei fugai")
                self.embeddings[user_id] = embedding
                self._save_embeddings()
                return True
            return False
        except Exception as e:
            logging.error(f"zuce shibai: {str(e)}")
            return False
    def recognize(self, face_roi):
        query_embed = self._get_embedding(face_roi)
        if query_embed is None:
            return None
        
        max_similarity = 0
        matched_user = None
        for user_id, saved_embed in self.embeddings.items():
            if len(saved_embed) != self.embedding_dim:
                logging.error(f"yonghu {user_id}tezhengtiquyichang:{len(saved_embed)}")
                continue
            
            similarity = cosine_similarity([query_embed], [saved_embed])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                matched_user = user_id
                
        return matched_user if max_similarity > self.threshold else None

    def _get_embedding(self, face_roi):
        try:
            #
            if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
                logging.warning("renlain quyu guoxiao")
                return None
            
            input_data = self._preprocess_face(face_roi)
            self.interpreter.set_tensor(
                self.input_details[0]['index'], 
                input_data
            )
            self.interpreter.invoke()
            
            embedding = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )[0]
            if len(embedding) != self.embedding_dim:
                raise ValueError(f"moxing shuchu weidu cuowu: {len(embedding)}")
            return embedding
            
        except Exception as e:
            logging.error(f"tezheng tiqu shibai :{str(e)}")
            return None
        #
        
        #return np.random.rand(128)
#