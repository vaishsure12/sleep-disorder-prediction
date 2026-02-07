from utils import SleepDisorder
from flask import Flask,jsonify,render_template,request
import config


app = Flask(__name__)

############################################  Home API  ##############################################################

@app.route('/')
def sleep_disorder_model():

    print('Welcome To The Homepage of Sleep Disorder model ')

    return render_template('home.html')
    # return 'TESTING OF SLEEP DISORDER MODEL' 

############################################  Model API  ##############################################################

@app.route('/predicted_disorder',methods= ['POST','GET'])

def get_sleep_disorder():
    if request.method == 'POST':
        print('We are in POST method')

        data = request.form
        Gender = data['Gender']
        Age = eval(data['Age'])
        Sleep_Duration = eval(data['Sleep_Duration'])
        Quality_of_Sleep = eval(data['Quality_of_Sleep'])
        Physical_Activity_Level = eval(data['Physical_Activity_Level'])
        Stress_Level = eval(data['Stress_Level'])
        BMI_Category = data['BMI_Category']
        Blood_Pressure = data['Blood_Pressure']
        Heart_Rate = eval(data['Heart_Rate'])
        Daily_Steps = eval(data['Daily_Steps'])
        Occupation  = data['Occupation']

        print(f'Gender={Gender},Age={Age},Sleep_Duration={Sleep_Duration},Quality_of_Sleep={Quality_of_Sleep},Physical_Activity_Level={Physical_Activity_Level},Stress_Level={Stress_Level},BMI_Category={BMI_Category},Blood_Pressure={Blood_Pressure},Heart_Rate={Heart_Rate},Daily_Steps={Daily_Steps},Occupation={Occupation}')
        
        sleep_dis = SleepDisorder(Gender,Age,Sleep_Duration,Quality_of_Sleep,Physical_Activity_Level,Stress_Level,BMI_Category,Blood_Pressure,Heart_Rate,Daily_Steps,Occupation)
        disorder = sleep_dis.get_predicted_disorder()
        if disorder[0] == 0:
            Result = 'This person is Healthy'
            return render_template('prediction.html',result=Result)
        elif disorder[0]==1:
            Result = 'This person is Suffering from Apnea'
            return render_template('prediction_apnea.html',result=Result)
        else:
            Result = 'This person is Suffering from Insomnia'
            return render_template('prediction_insomnia.html',result=Result)

if __name__=='__main__':
    app.run(host='127.0.0.1',port=config.PORT_NUMBER,debug=False)

     
            
         