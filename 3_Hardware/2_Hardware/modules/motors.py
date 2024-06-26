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
pwm_a.start(50)
pwm_b.start(50)

def decrease_speed():
    pwm_a.ChangeDutyCycle(30)
    pwm_b.ChangeDutyCycle(30)
    sleep(1)
    pwm_a.ChangeDutyCycle(20)
    pwm_b.ChangeDutyCycle(20)
    sleep(1)
    pwm_a.ChangeDutyCycle(10)
    pwm_b.ChangeDutyCycle(10)
    sleep(1)
    


# Function to move the wheelchair forward
def forward_motion():
    print("Moving forward...")
    decrease_speed()
    
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

    pwm_a.ChangeDutyCycle(40)
    pwm_b.ChangeDutyCycle(40)

# Function to move the wheelchair backward
def backward_motion(): 
    print("Moving backward...")
    decrease_speed()
    
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    
    pwm_a.ChangeDutyCycle(15)
    pwm_b.ChangeDutyCycle(15)
    sleep(1)
    pwm_a.ChangeDutyCycle(20)
    pwm_b.ChangeDutyCycle(20)
    sleep(1)
    pwm_a.ChangeDutyCycle(30)
    pwm_b.ChangeDutyCycle(30)

# Function to stop the motors
def stop_motion():
    print("Stopping motors...")
    decrease_speed()
    
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

# Function to move the wheelchair left
def left_motion():
    print("Turning left...")
    decrease_speed()
    
    GPIO.output(IN1, GPIO.HIGH)    
    pwm_b.ChangeDutyCycle(20)
    sleep(1)
    
    pwm_b.ChangeDutyCycle(0)


# Function to move the wheelchair right
def right_motion():
    print("Turning right...")
    decrease_speed()
    
    GPIO.output(IN3, GPIO.HIGH)

    pwm_a.ChangeDutyCycle(20)
    sleep(0.2)
    pwm_a.ChangeDutyCycle(0)
    

