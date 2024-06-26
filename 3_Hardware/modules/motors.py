import time
from time import sleep
import RPi.GPIO as GPIO

# Set the GPIO modes
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor 1
ENA = 25
IN1 = 23
IN2 = 24

# Motor 2
ENB = 22
IN3 = 17
IN4 = 27

# Setup pins for motors
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

# Initialize the pins
GPIO.output(IN1, GPIO.LOW)
GPIO.output(IN2, GPIO.LOW)
GPIO.output(IN3, GPIO.LOW)
GPIO.output(IN4, GPIO.LOW)

# Intialize PWM
pwm_a = GPIO.PWM(ENA, 1000)
pwm_b = GPIO.PWM(ENB, 1000)
pwm_a.start(40)
pwm_b.start(40)


# Function to stop the motors
def stop_motion():
    print("Stopping motors...")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)


# Function to move the wheelchair forward
def forward_motion():
    print("Moving forward...")
    
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    
    sleep(1)
    stop_motion()
    

# Function to move the wheelchair backward
def backward_motion(): 
    print("Moving backward...")
    
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    sleep(1)
    stop_motion()
    


# Function to move the wheelchair left
def left_motion():
    print("Turning left...")
    
    GPIO.output(IN1, GPIO.HIGH)  
    GPIO.output(IN2, GPIO.LOW)
      
    sleep(0.5)
    stop_motion()


# Function to move the wheelchair right
def right_motion():
    print("Turning right...")
    
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    sleep(0.5)
    stop_motion()
    



