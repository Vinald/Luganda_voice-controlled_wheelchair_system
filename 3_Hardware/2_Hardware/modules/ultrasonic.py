import RPi.GPIO as GPIO
from gpiozero import DistanceSensor


GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

ULTRASONIC_ECHO_PIN = 5
ULTRASONIC_TRIGGER_PIN = 6


def setup_ultrasonic_sensor(ULTRASONIC_ECHO_PIN, ULTRASONIC_TRIGGER_PIN):
    GPIO.setup(ULTRASONIC_ECHO_PIN, GPIO.IN)
    GPIO.setup(ULTRASONIC_TRIGGER_PIN, GPIO.OUT)

    ultrasonic_sensor = DistanceSensor(trigger=ULTRASONIC_TRIGGER_PIN, echo=ULTRASONIC_ECHO_PIN)
    GPIO.output(ULTRASONIC_TRIGGER_PIN, GPIO.LOW)

    def get_distance():
        return ultrasonic_sensor.distance * 100

    return get_distance
