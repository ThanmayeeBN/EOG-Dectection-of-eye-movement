#!interpreter
#C:\ProgramData\Anaconda3\python.exe
# -*- coding: utf-8 -*-


"""
{EOG Eye Movement Prediction}
{B.Tech Project}
"""

__author__ = 'sabarish'
__copyright__ = 'Copyright 2021, MAIN Part'
__credits__ = ['EnergeaTechnolabs']
__license__ = 'GNU'
__version__ = '1.0'
__maintainer__ = 'energeatechsolutions'
__email__ = 'info@energiasolutions.in'
__status__ = 'Completed'



from utilities import *
import matplotlib.pyplot as plt




if __name__ == "__main__":

    Data_Formatter = data_Formatter(filename="dataset.csv", debug = True)

    data_Frame = Data_Formatter.data_Reader()

    Data_Formatter.frame_Analyzer(data_F=data_Frame)

    data = Data_Formatter.EOG_data()

    Data_Processor = data_Preprocessor(data_Frame=data,debug=True)

    Prepared_Data = Data_Processor.data_Preparation()

    Data_Formatter.save_CSV(dataFrame = Prepared_Data,filename="prepared_Data.csv")

    print(Prepared_Data.dtypes)
    
    X,y,X_train,X_test,y_train,y_test = Data_Processor.data_Splitting()

    Prediction_Classifier = prediction_Using_Classifier(X_train = X_train,X_test = X_test,y_train = y_train,y_test = y_test,debug=True)

    Prediction_Classifier.create_Classifier(type_ = 1)

    clf = Prediction_Classifier.training_Classifier()

    Prediction_Classifier.testing_Classifier()

    a1,b1,c1 =Prediction_Classifier.evaluation_Criteria()
    
    #visualization(Prepared_Data,clf,X_train,X_test,y_train,y_test)

    Prediction_Classifier = prediction_Using_Classifier(X_train = X_train,X_test = X_test,y_train = y_train,y_test = y_test,debug=True)

    Prediction_Classifier.create_Classifier(type_ = 2)

    clf = Prediction_Classifier.training_Classifier()

    Prediction_Classifier.testing_Classifier()

    a2,b2,c2 = Prediction_Classifier.evaluation_Criteria()
    
    #visualization(Prepared_Data,clf,X_train,X_test,y_train,y_test)

    Prediction_Classifier = prediction_Using_Classifier(X_train = X_train,X_test = X_test,y_train = y_train,y_test = y_test,debug=True)

    Prediction_Classifier.create_Classifier(type_ = 3)

    clf = Prediction_Classifier.training_Classifier()

    Prediction_Classifier.testing_Classifier()

    a3,b3,c3 =Prediction_Classifier.evaluation_Criteria()
    
    #visualization(Prepared_Data,clf,X_train,X_test,y_train,y_test)

    Prediction_Classifier = prediction_Using_Classifier(X_train = X_train,X_test = X_test,y_train = y_train,y_test = y_test,debug=True)

    Prediction_Classifier.create_Classifier(type_ = 4)

    clf = Prediction_Classifier.training_Classifier()

    Prediction_Classifier.testing_Classifier()

    a4,b4,c4 =Prediction_Classifier.evaluation_Criteria()
    
    #visualization(Prepared_Data,clf,X_train,X_test,y_train,y_test)

    data = {'SVM':a1, 'DT':a2, 'RF':a3,'GB' :a4} 
    atypes = list(data.keys()) 
    values = list(data.values()) 
       
    fig = plt.figure(figsize = (10, 5)) 
      
    # creating the bar plot 
    plt.bar(atypes, values, color ='maroon',  
            width = 0.4) 
      
    plt.xlabel("Classifiers") 
    plt.ylabel("Accuracy Score") 
    plt.title("Accuracy Chart") 
    plt.show()

    data = {'SVM':b1, 'DT':b2, 'RF':b3,'GB' :b4} 
    atypes = list(data.keys()) 
    values = list(data.values()) 
       
    fig = plt.figure(figsize = (10, 5)) 
      
    # creating the bar plot 
    plt.bar(atypes, values, color ='maroon',  
            width = 0.4) 
      
    plt.xlabel("Classifiers") 
    plt.ylabel("Precision Score") 
    plt.title("Precision Chart") 
    plt.show()

    data = {'SVM':c1, 'DT':c2, 'RF':c3,'GB' :c4} 
    atypes = list(data.keys()) 
    values = list(data.values()) 
       
    fig = plt.figure(figsize = (10, 5)) 
      
    # creating the bar plot 
    plt.bar(atypes, values, color ='maroon',  
            width = 0.4) 
      
    plt.xlabel("Classifiers") 
    plt.ylabel("Recall Score") 
    plt.title("Recall Chart") 
    plt.show()



        
           

    
