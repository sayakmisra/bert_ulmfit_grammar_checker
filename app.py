from flask import Flask, request, jsonify, render_template
import bert
import ULMFIT

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    inp_string = [x for x in request.form.values()]
    sent = inp_string[0]
    index = bert.bert_checker(sent)
    index2 = ULMFIT.UFMFIT_checker(sent)[1]
    print("here1")
    output = "perfect" if index == 1 else "not right!!"
    output2 = "perfect" if index2 == 1 else "not right!!"
    return render_template('index.html', prediction_bert='BERT says: "{}" is grammatically {}'.format(sent, output),
                           prediction_ulmfit='ULMFIT says: "{}" is grammatically {}'.format(sent, output2))


if __name__ == "__main__":
    app.run(debug=True)
