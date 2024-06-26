import time
from time import sleep
from rpi_lcd import LCD
import RPi.GPIO as GPIO
from gpiozero import DistanceSensor
from signal import signal, SIGTERM, SIGHUP, pause


# Set the GPIO modes
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Personal packages
# For motor movements
from modules.motors import forward_motion, backward_motion, left_motion, right_motion, stop_motion

# For the buzzer
from modules.buzzer import buzzer_on_off

# Create an instance of the LCDModule
lcd = LCD()

try:
    lcd.clear()
    lcd.text('--> DAV SYSTEM <--', 2)
    lcd.text('------------------', 3)
    sleep(1)
    
    while True:
        #distance = int(get_distance())
        lcd.clear()
        lcd.text('--> DAV SYSTEM <--', 1)
        lcd.text('  TURN ON: GAALI', 2)
        
        print('DAV SYSTEM')
        command = input('Enter a command \n m-mumaaso | e-emabega \n k-kkono    | d-ddyo \n y-yimirira    | q-Exit the system \n').lower()
        
        if command == 's':
            lcd.clear()
            lcd.text('   DAV SYSTEM ON', 1)
            lcd.text('  Stopping.....', 2)
            stop_motion()
        else:        
            if command == 'w':
                lcd.clear()
                lcd.text('   DAV SYSTEM ON', 1)
                lcd.text('  Moving forward', 2)
                
                forward_motion()
                
            elif command == 'x':
                lcd.clear()
                lcd.text('   DAV SYSTEM ON', 1)
                lcd.text('  Moving Backward', 2)
                
                backward_motion()
                
            elif command == 'a':
                lcd.clear()
                lcd.text('   DAV SYSTEM ON', 1)
                lcd.text('  Moving Left', 2)
                
                left_motion()
            elif command == 'd':
                lcd.clear()
                lcd.text('   DAV SYSTEM ON', 1)
                lcd.text('  Moving Right', 2)
                
                right_motion()
                
            elif command == 'q':
                lcd.clear()
                lcd.text('   DAV SYSTEM OFF', 1)
                lcd.text('  Exiting System', 2)
                
                stop_motion()
                
            else:
                lcd.clear()
                lcd.text(' Invalid wakeword', 2)
                continue


except KeyboardInterrupt:
    lcd.clear()
    lcd.text('Wheelchair off', 1)
    GPIO.cleanup()
    print("Program exited successfully.")


finally:
    lcd.clear()
    lcd.text('Wheelchair off', 1)
    GPIO.cleanup()
    print("Program exited successfully.")

