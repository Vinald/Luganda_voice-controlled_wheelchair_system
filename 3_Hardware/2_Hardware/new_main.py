import time
from time import sleep
import RPi.GPIO as GPIO
from rpi_lcd import LCD
from gpiozero import DistanceSensor
from signal import signal, SIGTERM, SIGHUP, pause


# Set the GPIO modes
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Set the GPIO pin constants
# Buzzer
BUZZER_PIN = 4

# Ultrasonic sensor
ULTRASONIC_ECHO_PIN = 5
ULTRASONIC_TRIGGER_PIN = 6

# Motor 1
ENA = 25
IN1 = 23
IN2 = 24

# Motor 2
ENB = 22
IN3 = 17
IN4 = 27

# Set up the GPIO pins
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(ULTRASONIC_ECHO_PIN, GPIO.IN)
GPIO.setup(ULTRASONIC_TRIGGER_PIN, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
# GPIO.setup(EMERGENCY_STOP_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Initialize the pins
GPIO.output(BUZZER_PIN, GPIO.LOW)
GPIO.output(ULTRASONIC_TRIGGER_PIN, GPIO.LOW)
GPIO.output(IN1, GPIO.LOW)
GPIO.output(IN2, GPIO.LOW)
GPIO.output(IN3, GPIO.LOW)
GPIO.output(IN4, GPIO.LOW)

# Intialize PWM
pwm_a = GPIO.PWM(ENA, 1000)
pwm_b = GPIO.PWM(ENB, 1000)
pwm_a.start(0)
pwm_b.start(0)

# Initialize the LCD
lcd = LCD()

# Initialize the ultrasonic sensor
ultrasonic_sensor = DistanceSensor(trigger=ULTRASONIC_TRIGGER_PIN, echo=ULTRASONIC_ECHO_PIN)


# Function to exit the program safely
def safe_exit(signum, frame):
    exit(1)


# Function to print the distance
def print_distance(distance):
    lcd.text(f"Distance: {distance:.2f})", 2)



try:
    # Print on the lcd
    lcd.text("WHEELCHAIR ON", 1)
    lcd.text("f-start | b-back", 2)
    lcd.text("l-left  | r-right", 3)
    lcd.text("s-stop  | e-exit", 4)
    
    while True:
        
        distance = ultrasonic_sensor.distance * 100
        print(f"Distance: {distance:.2f} cm")

        # Get the command from the user
        command = input('Enter the command \n f-forward \n b-backward \n l-left \n r-right \n s-stop \n e-exit \n')
        print(f"Command entered: {command}")

        if command == 'f':
            forward_motion()
            lcd.clear)()
            lcd.text("Moving forward", 1)
            print_distance(distance)
            sleep(1)
            
            if (distance < 20):
                GPIO.output(BUZZER_PIN, GPIO.HIGH)
                stop_motion()
                sleep(3)
                GPIO.output(BUZZER_PIN, GPIO.LOW)
            else:
                GPIO.output(BUZZER_PIN, GPIO.LOW)

        elif command == 'b':
            backward_motion()
            lcd.clear()
            lcd.text("Moving backward", 1)
            print_distance(distance)
            sleep(1)

        elif command == 'l':
            left_motion()
            lcd.clear()
            lcd.text("Moving left", 1)
            print_distance(distance)
            sleep(1)

        elif command == 'r':
            right_motion()
            lcd.clear()
            lcd.text("Moving right", 1)
            print_distance(distance)
            sleep(1)

        elif command == 's':
            stop_motion()
            lcd.clear()
            lcd.text("Stopping", 1)
            print_distance(distance)
            sleep(1)

        elif command == 'e':
            stop_motion()
            lcd.clear()
            lcd.text("Exiting", 1)
            pwm_a.stop()
            pwm_b.stop()
            GPIO.cleanup()
            break

        else:
            print("Invalid command. Please try again.")
            continue

except KeyboardInterrupt:
    lcd.clear()
    lcd.text('Wheelchair off', 1)
    pwm_a.stop()
    pwm_b.stop()
    GPIO.cleanup()
    print("Program exited successfully.")

finally:
    lcd.clear()
    lcd.text('Wheelchair off', 1)
    pwm_a.stop()
    pwm_b.stop()
    GPIO.cleanup()
    print("Program exited successfully.")