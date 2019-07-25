from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from joblib import dump, load
import operator
import os
#Divicion de los datos
from sklearn.model_selection import train_test_split
from mfcc_data_audio import MFCCData as mfcc
#Random forest
from sklearn.ensemble import RandomForestClassifier
import pickle

class NeuralNetSl:

    __MODELS_PATH = "resources" + os.path.sep + "models" + os.path.sep + "neuralnetsl"

    def __init__(self,labels=[]):
        self.__encoder = LabelEncoder()
        self.__encoder = self.__encoder.fit(labels)
        self.__classes = self.__encoder.classes_


    def encode_labels(self, labels):
        return self.__encoder.transform(labels)



    def get_classes(self):
        return self.__classes.tolist()

    #20,10 saca el 100%
    def build_model(self, seed=None):
        return MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(500, 250,70,20), random_state=1)

    def fit_model(self, x, y, model):
        return model.fit(x, y)

    def load_model(self, id):
        path = self.__MODELS_PATH + os.path.sep + id
        if os.path.exists(path):
            file_name = path + os.path.sep + id + ".joblib"
            return load(file_name)

    def save_model(self, id, model):
        path = self.__MODELS_PATH + os.path.sep + id
        if not os.path.exists(path):
            os.makedirs(path)
            file_name = path + os.path.sep + id + ".joblib"
            dump(model, file_name)

    def predict(self, x, model):
        results = model.predict_proba(x)
        predictions_prob = []
        predictions = []

        for result in results:
            prediction_prob = dict(zip(self.get_classes(), result))
            prediction = max(prediction_prob.items(),
                             key=operator.itemgetter(1))[0]
            predictions_prob.append(prediction_prob)
            predictions.append(prediction)
        return predictions, predictions_prob

    def evaluate(self, x, y, model):
        return model.score(x, y)

    def confusion_matrix(self, y_true, y_pred, show=False):
        matrix = confusion_matrix(y_true, y_pred)
        if (show):
            print("\t", self.get_classes())
            for x in range(len(matrix)):
                print(str(self.get_classes()[x]) + "\t", matrix[x])

        return matrix
    #Parte de random forest
    def buildmodel_random_forest(self, X,y):
        model_filename="resources/models/model_random/finalized_model.sav"
        clf=RandomForestClassifier(n_estimators=100)
        X_train, X_test, y_train, y_test = train_test_split(X,y)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        #Guardar el modelo de arboles de desicion
        pickle.dump(clf, open(model_filename, 'wb'))
        # fin del guardado
        print(self.evaluate(X_test,y_test,clf))
        self.confusion_matrix(y_test, y_pred, True)
        #predictions, predictions_prob = self.predict([mfcc().extraer_MFCC("Perro llorando de dolor.wav")],clf)
        #print("\nPrediction:", predictions[0], predictions_prob[0])

    def load_model_random(self):
        model_filename="resources/models/model_random/finalized_model.sav"
        # load the model from disk
        loaded_model = pickle.load(open(model_filename, 'rb'))
        return loaded_model    	
class predict:
	def __init__(self):
		print("inicia la prediccion")
	def predecir(self,path_archivo,modeltype="RN"):
		X, y = mfcc().crear_dataCSV()
		X_train, X_test, y_train, y_test = train_test_split(X,y)
		nn = NeuralNetSl( y)
		if modeltype=="RN":
			model=nn.load_model("nn_sl_model1")
			predictions, predictions_prob = nn.predict([mfcc().extraer_MFCC(path_archivo)], model)
			evaluar=nn.evaluate(X_test,y_test,model)
		elif modeltype=="RF":
			modelRF=nn.load_model_random()
			predictions, predictions_prob = nn.predict([mfcc().extraer_MFCC(path_archivo)],modelRF)
			evaluar=nn.evaluate(X_test,y_test,modelRF)
		return predictions[0], predictions_prob[0],evaluar




#MAI
if __name__ == "__main__":
    X, y = mfcc().crear_dataCSV()
    nn = NeuralNetSl(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    #model = nn.build_model()
    #model = nn.fit_model(X_train, y_train, model)

    #nn.save_model("nn_sl_model1", model)
    nn.buildmodel_random_forest(X,y)
    model = nn.load_model("nn_sl_model1")

    y_pred = model.predict(X_test)

    nn.confusion_matrix(y_test, y_pred, True)
    print(nn.evaluate(X_test,y_test,model))

    #predictions, predictions_prob = nn.predict([mfcc().extraer_MFCC("pajaro.wav")], model)
    #print("\nPrediction:", predictions[0], predictions_prob[0])
    #print("Prediction Prob:", predictions_prob[0])

    
    #print(predict().predecir("Perro llorando de dolor.wav"))


