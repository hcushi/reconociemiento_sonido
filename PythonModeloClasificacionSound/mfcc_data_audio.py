
from python_speech_features import mfcc
from python_speech_features import logfbank
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy
import pydub
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
import librosa
import soundfile as sf
import re

class MFCCData:
	#Variables de direccion
	#__DATA_AUDIO_FILE(contiene  en nombre de los archivos de audio)
	#__PATH_AUDIO_FILE(establece la direccion de los archivos de audio)
	#__POLARITY_FILE(contiene la polaridad para el entrenamiendo )
	#__TRAIN_FILE(contiene la data de 13 variables extraidas de los archivos de audio)
	__DATA_AUDIO_FILE="/resources/data/data_audio.csv"
	__PATH_AUDIO="resources/data/"
	__POLARITY_FILE="resources/training/polaridad.csv"
	__TRAIN_FILE="resources/training/training.csv"

	def __init__(self):
		print("inicio")


		
	#Permite extraer la huella con MFCC del sonido mediante 13 variables que caracterizan al sonido
	
	def extract_mfcc13(self,path_sound):
		(sig,rate)=sf.read(path_sound)
		#(sig, rate) =librosa.load(path_sound, sr=None)
		#(rate,sig)=wav.read(self.__PATH_AUDIO + path_sound)
		mfcc_feat=mfcc(sig,rate,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=1103,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)  
		z= self.PCAOptimizado(mfcc_feat,1)
		return  z

	def extencion(self, path_archivo):
		nombre,extencion=os.path.splitext(path_archivo)
		return nombre,extencion


	#Extrae las 13 variables de la matriz de MFCC(Muy demoros)
	def pca(self,x,k):
		covar_x = np.dot(x.T,x)/x.shape[0]
		[U,S,V] = scipy.linalg.svd(covar_x)
		Z = np.dot(x,U[:,0:k])
		z_data= np.array(Z.T)[0]
		return z_data

	#vector x del sonido , k componentes principales o la huella del audio
	def PCAOptimizado(self, x,k):
		flattend_mfcc = np.array(x)
		flattend_mfcc = flattend_mfcc.transpose()
		pca = PCA(n_components=k)
		pca.fit(flattend_mfcc)
		sample = pca.transform(flattend_mfcc)
		return np.array(sample.T)[0]

	#Recive un archivo csv con el Path del audio el preditivo
	#Ejemplo: '/resource/audio01.wav','GATO'
	#Devuelve arreglos con el path de audio y polaridad
	# Lee el archivo csv que tiene la lista de entrenamiento
	# return: dataframe
	def readFile(self, file_path):
	    return pd.read_csv(file_path, sep=";",encoding = "ISO-8859-1")

	#Permite generar la data de traing de un archivo csv con "path";"polaridad"
	#Devuelve dos listas (Polaridad(y) y traning(x)) 
	#crea los archivos de traning y polaridad

	def save_data(self):
		traning_data=[]
		polarity_data=[]
		pathAudios=self.readFile(self.__DATA_AUDIO_FILE)["path"]
		polaridad=self.readFile(self.__DATA_AUDIO_FILE)["polaridad"]
		#guardar el archivo de polaridad
		if os.path.exists(self.__TRAIN_FILE)==False:
			row_polaridad=np.array(['polaridad'])
			np.savetxt(self.__POLARITY_FILE,np.hstack((row_polaridad,polaridad)) ,delimiter=';',fmt='%s')
			polarity_data= np.array(polaridad.T)
			for audio_path in pathAudios:
				print("Guardado el MFFCC de ", audio_path)
				traning_data.append(self.extract_mfcc13(self.__PATH_AUDIO +audio_path).tolist())
			#print(polarity_data.tolist(), traning_data)
			#guardar el archivo de traning
			np.savetxt(self.__TRAIN_FILE,traning_data,delimiter=';',header="v1;v2;v3;v4;v5;v6;v7;v8;v9;v10;v11;v12;v13")
		x=self.readFile(self.__TRAIN_FILE)
		y=self.readFile(self.__POLARITY_FILE)
		return x,y


	def crear_dataCSV(self,load_path=__PATH_AUDIO):
		# list load_path sub-folders
		regex = re.compile(r'^[0-9]') # to detect  sound file folders starting with numerals
		directory_list = [i for i in os.listdir(load_path) if regex.search(i)]
		# initialize empty data frame for results
		concat_features = pd.DataFrame()
		polaridad_data=[]
		polaridad_dataset=[]
		traning_data=[]
		# iteration on sub-folders
		if os.path.exists(self.__TRAIN_FILE)==False:
			for directory in directory_list:
				polaridad_data.append(directory)
				file_list = os.listdir(os.path.join(load_path, directory))
				for audio_file in file_list:
					print (os.path.join(load_path, directory, audio_file))
					polaridad_dataset.append(directory)
					traning_data.append(self.extract_mfcc13(os.path.join(load_path, directory, audio_file)).tolist())
				np.savetxt(self.__TRAIN_FILE,traning_data,delimiter=';',header="v1;v2;v3;v4;v5;v6;v7;v8;v9;v10;v11;v12;v13")
				print(directory)
			row_polaridad=np.array(['polaridad'])
			np.savetxt(self.__POLARITY_FILE,np.hstack((row_polaridad,polaridad_dataset)) ,delimiter=';',fmt='%s')
		x=self.readFile(self.__TRAIN_FILE)
		y=self.readFile(self.__POLARITY_FILE)
		return x,y

    


	def extraer_MFCC(self,audio):
		return self.extract_mfcc13(audio).tolist()

	#Grafica la senial
	def graficarSenial(self,x):
		plt.plot(x)
		plt.show() 

#mfccgr=MFCCData()
#archivo= "1-34094-B.ogg"
#(sig,rate)=sf.read(archivo)
#mfcc_feat=mfcc(sig,rate,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=1103,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)  
#mfccgr.graficarSenial(sig)


#(rate,sig)=wav.read("ArjonaOriginal.wav")
#mfcc_feat=mfcc(sig,rate,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=1103,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)  
#flattend_mfcc = np.array(mfcc_feat)
#flattend_mfcc = flattend_mfcc.transpose()
#pca = PCA(n_components=1)
#pca.fit(flattend_mfcc)
#sample = pca.transform(flattend_mfcc)
#print(np.array(sample.T)[0])

#MFCC=MFCCData()
#x,y=MFCC.save_data()  
#print(x)
#print(MFCC.extraer_MFCC("elefante.wav"))
#x,y=MFCC.crear_dataCSV()
#print(x,y)