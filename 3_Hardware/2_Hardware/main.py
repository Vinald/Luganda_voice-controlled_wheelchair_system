from time import sleep
from rpi_lcd import LCD
import RPi.GPIO as GPIO
# from signal import signal, SIGTERM, SIGHUP, pause

# For motor movements
from modules.motors import forward_motion, backward_motion, left_motion, right_motion, stop_motion
from modules.buzzer import buzzer_on_off
from modules.ultrasonic import setup_ultrasonic_sensor

# Set the GPIO modes
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Create an instance of the LCDModule
lcd = LCD()

try:
    lcd.clear()
    lcd.text('--> A&S SYSTEM <--', 2)
    lcd.text('-------------------', 3)
    sleep(1)
    
    while True:
        lcd.clear()
        lcd.text('--> SYSTEM ON <--', 1)
        lcd.text('  Say Wakeword', 2)
        lcd.text(' To Start System', 3)
        
        wakeword = input('Enter the wakeword (g, G)\n').lower()
        
        if wakeword == 'g':
            lcd.clear()
            lcd.text('  A&S SYSTEM ON', 1)
            lcd.text(' MUMAASO | EMABEGA', 2)
            lcd.text(' <- KKONO DDYO ->', 3)
            lcd.text('    YIMIRIRA  ', 4)

            
            while True:
                #distance = int(get_distance())
                
                print('DAV SYSTEM')
                command = input('Enter a command \n m-mumaaso | e-emabega \n k-kkono    | d-ddyo \n y-yimirira    | q-Exit the system \n').lower()
                
                if command == 'w':
                    #if distance < 10:
                        #lcd.clear()
                        #lcd.text('Object detected at',1)jup
                        #lcd.text(f'{distance} cm',2)
                        
                        #buzzer_on_off()
                        #stop_motion()
                        
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
                    
                elif command == 's':
                    lcd.clear()
                    lcd.text('   DAV SYSTEM ON', 1)
                    lcd.text('  Stopping.....', 2)
                    
                    stop_motion()
                    
                elif command == 'q':
                    lcd.clear()
                    lcd.text('   DAV SYSTEM OFF', 1)
                    lcd.text('  Exiting System', 2)
                    
                    stop_motion()
                    
                else:
                    lcd.clear()
                    lcd.text(' Invalid wakeword', 2)
                    continue
                
        else:
            lcd.clear()
            lcd.text(' Invalid wakeword', 2)
            sleep(1)
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
