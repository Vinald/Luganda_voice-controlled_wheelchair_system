# Luganda Voice Controlled Wheelchair System

This project is a voice-controlled wheelchair system that uses Luganda language commands. The following are the commands and statements used for data collection.

## How to run the code

1. `Create a virtual environment`
2. `Run the requirements.txt file`
3. `Depending on the audio features you plan to use`

    - `If Mel-spectrograms skip step 4, but if using MFCC run step 4`

4. `Run the mfcc_extraction file in packages to extract mfccs and create json files`

### MFCC extraction

- Navigate to `packages/mfcc.py`
- Run the python file to create json files in the json folder of the extracted MFCCs

## Commands

- `mu masso`: Moves the wheelchair forward
- `emabega`: Moves the wheelchair backward
- `ddyo`: Turns the wheelchair to the right
- `kkono`: Turns the wheelchair to the left
- `yimirira`: Stops the wheelchair
- `gaali`: Wake word to activate the system

### Left

The following statements are used to command the wheelchair to turn left:

- `kkono`, `dda ku kkono`, `ku kkono`, `kyukira ku kkono`, `weta ku kkono`

### Right

The following statements are used to command the wheelchair to turn right:

- `ddyo`, `dda ku ddyo`, `ku ddyo`, `kyukira ku ddyo`, `weta ku ddyo`

### Forward

The following statements are used to command the wheelchair to move forward:

- `mu maaso`, `dda mu maaso`, `mu maaso awo`, `weeyongere mu maaso`, `genda mu maaso awo`

### Backward

The following statements are used to command the wheelchair to move backward:

- `emabega`, `dda emabega`, `emabega awo`, `okudda emabega`, `genda emabega`

### Stop

The following statements are used to command the wheelchair to stop:

- `yimirira`, `okuyimirira`, `yimirira awo`, `gwe yimirira`, `kaati yimirira`

### Wake-Word

The following statement is used as the wake word to activate the system:

- `gaali`
