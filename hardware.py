import time
import logging
import cv2
from gpiozero import LED, Buzzer
from picamera2 import Picamera2

class OV5647Controller:
    def __init__(self):
        self.led = LED(17)        
        self.buzzer = Buzzer(27)          
        self.picam2 = Picamera2(camera_num=0)   
        self._setup_camera()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("OV5647chishihua wac")

    def _setup_camera(self):
        
        config = self.picam2.create_still_configuration(
            #main={"size": (2592, 1944)},1296x972
            #main={"size": (1311, 1860)},
            main={"size": (1296, 972)},
            #main={"size": (640, 480)},
            buffer_count=4
        )
        self.picam2.configure(config)
        self.picam2.set_controls({
            "AwbMode": 0,
            "AeEnable": True,
            "AeExposureMode": 1, 
            #"FrameDurationLimits": (33333,33333),
            "AnalogueGain": 5.5
        })
        self.picam2.start()
        time.sleep(2)  

    def capture_frame(self):
        try:
            frame = self.picam2.capture_array("main")
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#             return frame

        except Exception as e:
            logging.error(f"puhu yichang: {str(e)}")
            return None

    def led_success(self):
        self.led.blink(on_time=0.3, off_time=0.3, n=3)
        
    def buzzer_alert(self):
        self.buzzer.beep(on_time=0.5, off_time=0.2, n=2)

    def cleanup(self):
        self.picam2.stop()
        self.led.off()
        self.buzzer.off()
        logging.info("yingian shifang")