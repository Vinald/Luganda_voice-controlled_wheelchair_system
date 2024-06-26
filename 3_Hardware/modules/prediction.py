from time import sleep


# Function to process the predicted label
def process_predicted_label(predicted_label):
    while True:

        print('DAV SYSTEM')
        print('Say the command')
        
        if predicted_label == 'mumaaso':
            print('Moving Forward')
            sleep(2)
            break
        elif predicted_label == 'emabega':
            print('Moving Backward')
            sleep(2)
            break
        elif predicted_label == 'kkono':
            print('Moving Left')
            sleep(2)
            break
        elif predicted_label == 'ddyo':
            print('Moving Right')
            sleep(2)
            break
        elif predicted_label == 'yimirira':
            print('Stopping.....')
            sleep(2)
            break
            
        else:
            print(' Invalid wakeword')
            continue
