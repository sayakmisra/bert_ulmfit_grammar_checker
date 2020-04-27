from fastai.text import *

def UFMFIT_checker(sent):
    print("inside ULMFIT:")
    # Path of the folder where we will have the saved model.
    learn = load_learner('/Users/sayak/projects/gram_checker/Heroku-Demo/model_ulmfit')
    index = learn.predict(sent)
    return index