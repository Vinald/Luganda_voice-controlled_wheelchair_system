import tensorflow as tf 
import numpy as np 
import time
import subprocess

# Function to load TFlite model
def loadTfliteModel(modelPathIsPlacedHere):
	interpreter = tf.lite.Interpreter(modelPath= modelPathIsPlacedHere)
	interpreter.allocate_tensors()
	return interpreter

# Function to perform inference
def inference (interpreter, inputData):
	inputDetails = interpreter.get_input_details()
	outputDetails = interpreter.get_output_details()

	interpreter.set_tensor(inputDetails[0]['index'], inputData)
	interpreter.invoke()
	outputData = interpreter.get_tensor(outputDetails[0]['index'])

	return outputData

#load TFlite models

speech_intent_model = loadTfliteModel(path/to/model...)
wake_word_model = loadTfliteModel(path/to/model...)


#function to 

def main():
	while True:
		audio_data = capture_audio()
		test_speech_intent_classifier(audio_data)
		test_wake_word_detector(audio_data)
		time.sleep(1)

if __name__ == "__main__":

	main()

	