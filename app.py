from flask import Flask, render_template, request
from flask_restful import reqparse, abort, Api, Resource
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# instantiating Flask RESTFul API
app = Flask(__name__)
api = Api(app)

# loading model
model = joblib.load('model.pkl')

# argument parsing
try:
    parser = reqparse.RequestParser()
    parser.add_argument('idade', type=float)
    parser.add_argument('esv', type = float, action='append')
    parser.add_argument('essv', type = float, action='append')
    parser.add_argument('imve', type = float, action='append')
    parser.add_argument('vae', type = float, action='append')
    parser.add_argument('u', type = float, action='append')
    parser.add_argument('creat', type = float, action='append')
    parser.add_argument('k', type = float, action='append')
    parser.add_argument('ct', type = float, action='append')
except:
    print('no request made yet')

class PredictProbability(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()

        #user_query1 = args['idade']
        #user_query2 = args['esv']
        #user_query3 = args['essv']
        #user_query4 = args['imve']
        #user_query5 = args['vae']
        #user_query6 = args['u']
        #user_query7 = args['creat']
        #user_query8 = args['k']
        #user_query9 = args['ct']
        try:
            idade = float(request.args.get('idade'))
            esv = float(request.args.get('esv'))
            essv = float(request.args.get('essv'))
            imve = float(request.args.get('imve'))
            vae = float(request.args.get('vae'))
            u = float(request.args.get('u'))
            creat = float(request.args.get('creat'))
            k = float(request.args.get('k'))
            ct = float(request.args.get('ct'))
        except:
            pass

        # converting to np.array
        #user_query1 = np.asarray(user_query1)
        #user_query2 = np.asarray(user_query2)
        #user_query3 = np.asarray(user_query3)
        #user_query4 = np.asarray(user_query4)
        #user_query5 = np.asarray(user_query5)
        #user_query6 = np.asarray(user_query6)
        #user_query7 = np.asarray(user_query7)
        #user_query8 = np.asarray(user_query8)
        #user_query9 = np.asarray(user_query9)

        # run predictions on the array of features
        try:
            #pred_proba = model.predict_proba(np.array([user_query1, user_query2,
             #   user_query3, user_query4, user_query5, user_query6,
              #  user_query7, user_query8, user_query9]).reshape(-1, 9))[:,1]
            pred_proba = model.predict_proba(np.array([idade, esv,
                essv, imve, vae, u,
                creat, k, ct]).reshape(-1, 9))[:,1]


            # round the predict proba value and set to new variable
            var = int(pred_proba*100)

            # create JSON object
            print(var)
            output = {'prediction': str(var)+'%'}
            
            return output
        except:
            return 'Request already not made'

# Setup the Api resource routing here
# Route the URL to the resource

api.add_resource(PredictProbability, '/')


if __name__ == '__main__':
    app.run(debug=True)
