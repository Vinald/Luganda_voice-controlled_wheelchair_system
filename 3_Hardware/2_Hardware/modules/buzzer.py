from time import sleep
import RPi.GPIO as GPIO


GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

BUZZER_PIN = 4   


def buzzer_on_off(buzzer_pin=BUZZER_PIN):
    GPIO.setup(buzzer_pin, GPIO.OUT)

    try:
        GPIO.output(buzzer_pin, GPIO.HIGH)
        print("Buzzer on")
        sleep(1)

        GPIO.output(buzzer_pin, GPIO.LOW)
        print("Buzzer off")
        sleep(1)

    finally:
        GPIO.cleanup()