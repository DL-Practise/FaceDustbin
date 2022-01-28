import RPi.GPIO as GPIO
import time

duoji = 40
GPIO.setmode(GPIO.BOARD)
GPIO.setup(duoji, GPIO.OUT, initial=False)
pwm = GPIO.PWM(duoji, 50)
pwm.start(0)
pwm.ChangeDutyCycle(10)
time.sleep(1)
pwm.stop()
GPIO.cleanup()

