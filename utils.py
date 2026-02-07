import pickle
import json
import numpy as np



class SleepDisorder():

    def __init__(self,Gender,Age,Sleep_Duration,Quality_of_Sleep,Physical_Activity_Level,Stress_Level,BMI_Category,Blood_Pressure,Heart_Rate,Daily_Steps,Occupation):
        self.Gender = Gender
        self.Age = Age
        self.Sleep_Duration = Sleep_Duration
        self.Quality_of_Sleep = Quality_of_Sleep
        self.Physical_Activity_Level = Physical_Activity_Level
        self.Stress_Level = Stress_Level
        self.BMI_Category = BMI_Category
        self.Blood_Pressure = Blood_Pressure
        self.Heart_Rate = Heart_Rate
        self.Daily_Steps = Daily_Steps
        self.Occupation = 'Occupation_' + Occupation

    def load_model(self):
        with open('project_app/Logistic_model.pkl','rb') as f:
            self.model = pickle.load(f)

        with open('project_app/project_data.json','rb') as f:
            self.project_data = json.load(f)

    def get_predicted_disorder(self):
        self.load_model()

        test_array = np.zeros(len(self.project_data['columns']))
        test_array[0] = self.project_data['Gender'][self.Gender]
        test_array[1] = self.Age
        test_array[2] = self.Sleep_Duration
        test_array[3] = self.Quality_of_Sleep
        test_array[4] = self.Physical_Activity_Level
        test_array[5] = self.Stress_Level
        test_array[6] = self.project_data['BMI Category'][self.BMI_Category]
        test_array[7] = self.project_data['Blood Pressure'][self.Blood_Pressure]
        test_array[8] = self.Heart_Rate
        test_array[9] = self.Daily_Steps
        Occupation_index = self.project_data['columns'].index(self.Occupation)
        test_array[Occupation_index] = 1

        print('Test Array',test_array)

        predicted_disorder = self.model.predict([test_array])
        print(f'This person is suffering from sleep disorder category=={predicted_disorder[0]}')
        return predicted_disorder
    

if __name__ == '__main__':
    Gender = 'Male'
    Age   = 26
    Sleep_Duration = 4
    Quality_of_Sleep = 7
    Physical_Activity_Level = 44
    Stress_Level = 7
    BMI_Category = 'Overweight'
    Blood_Pressure = 'High Blood Pressure (Stage 1)'
    Heart_Rate = 80
    Daily_Steps = 4000
    Occupation  = 'Doctor'

    sleep_dis = SleepDisorder(Gender,Age,Sleep_Duration,Quality_of_Sleep,Physical_Activity_Level,Stress_Level,BMI_Category,Blood_Pressure,Heart_Rate,Daily_Steps,Occupation)
    sleep_dis.get_predicted_disorder()

            