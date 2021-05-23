## SECTION 1 : PROJECT TITLE
## Intelligent Sensing System Project - Gym Buddies

![GymBuddies](./img.jpeg)

---

## SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT
Incorrect form is one of the leading factors of injury when performing exercises. This project aims to help users identify mistakes via real-time or with a pre-recorded video when performing squats or push-ups. This is done via a visual feedback made possible by a combination of methods from pose-estimation, 3D-CNNs, statistical approaches and a rules engine with an input of a pair of synchronised monochrome and disparity videos. The visual feedback provides information on the exercise being performed, the number of completed repetitions, a highlight of the users pose where mistakes are made and the number of failed repetitions. The interface is available via both command-line as well as a web-interface.

---

## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

| Name  | Student ID  | Work Items | Email |
| :------------ |:---------------:| :-----| :-----|
| Mohamed Mikhail Kennerley | A0213546J | • Pose Feature Extraction <br>• Mono-D Dataset <br>• 3DCNN Classifier| e0508649@u.nus.edu |
| Vidish Metha | A0213523U | • Repetition Counting <br>• LBP-SVM / 2DCNN <br>•  Web Interface| e0508624@u.nus.edu|
| Oh Chun How | A1234567C | • Rules Engine <br>• HoG-SVM <br>• Initial Dataset| Chunhow.oh@u.nus.edu |

---

## SECTION 4 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO

[![GymBuddies](https://img.youtube.com/vi/WCCdBEJB0-Y/0.jpg)](https://youtu.be/WCCdBEJB0-Y "
GymBuddies")
---

## SECTION 5 : USER GUIDE
**Step 1**: Download videos at: https://drive.google.com/drive/folders/15cHzsHID29E2F2-_q_6-2ybPo4frhe32?usp=sharing <br>
**Step 2**: Create a new conda environment with conda_requirements.txt <br>
*$ conda create --name env --file conda_requirements.txt* <br>
**Step 3**: Install other requirements via pip <br>
*$ pip install -r pip_requirements.txt* <br>
**Step 4**: In visionSysPre folder run: <br>
*$ python main.py --mono "Mono video file" --depth "Depth video file"*

#### Frontend-Interface

**Step 1**: Create a new conda environment with web_requirements.txt (within GymBuddyWeb folder)<br>
*$ conda create --name env --file conda_requirements.txt* <br>
**Step 2**: Collect the static files<br>
*$ python manage.py collectstatic* <br>
**Step 3**: Serve the web interface in local host<br>
*$ python manage.py runserver*


---
## SECTION 6 : PROJECT REPORT / PAPER

Refer to project report: Report-GymBuddies <br>
