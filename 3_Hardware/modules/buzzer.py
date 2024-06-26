from time import sleep
import RPi.GPIO as GPIO

# Set the GPIO modes
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

BUZZER_PIN = 4

def buzzer_on_off(buzzer_pin=BUZZER_PIN):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(buzzer_pin, GPIO.OUT)

    GPIO.output(buzzer_pin, GPIO.HIGH)
    print("Buzzer on")
    sleep(1)

    GPIO.output(buzzer_pin, GPIO.LOW)
    print("Buzzer off")
    sleep(1)

