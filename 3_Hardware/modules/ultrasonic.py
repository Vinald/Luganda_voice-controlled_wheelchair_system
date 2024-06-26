from gpiozero import DistanceSensor
import RPi.GPIO as GPIO

# Set the GPIO modes
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Ultrasonic sensor
ULTRASONIC_ECHO_PIN = 5
ULTRASONIC_TRIGGER_PIN = 6

# Initialize the ultrasonic sensor
ultrasonic_sensor = DistanceSensor(trigger=ULTRASONIC_TRIGGER_PIN, echo=ULTRASONIC_ECHO_PIN)

# Set up the GPIO pins
GPIO.setup(ULTRASONIC_ECHO_PIN, GPIO.IN)
GPIO.setup(ULTRASONIC_TRIGGER_PIN, GPIO.OUT)

# Initialize the pins
GPIO.output(ULTRASONIC_TRIGGER_PIN, GPIO.LOW)


def get_distance():
    return ultrasonic_sensor.distance * 100
