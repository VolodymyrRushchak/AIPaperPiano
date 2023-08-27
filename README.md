# AIPaperPiano

https://github.com/VolodymyrRushchak/AIPaperPiano/assets/93164951/c03d4b7f-098c-4b01-8306-e1daf764304f

## Description
AIPaperPiano allows you to play on a virtual piano. All you need is a laptop with a web camera and two A4 sheets (and a table of course:). <br />
Here I used the Python programming language, Tensorflow, NumPy and Seaborn libraries, and the PyQt5 graphics library. <br />
During the development of the project I learned and practiced such concepts as transfer learning, fine tuning, data augmentation, hyperparameters tuning, active learning, learning rate scheduler, etc. I also had experience with Google Colab (used it for training the final model).

## Project's Structure
As you can see the project is divided into five directories: 
1. piano_interface: This is the main directory, which contains the final application. If you just want to play piano, all you need is in this and the next one folders.
2. piano_templates: Here you can find pdf files with the two parts of a paper piano. You need to print them and put them on your table in the right order.
3. aimodel: This directory contains the code I used for the model training.
4. data_collector: In this directory you can find an application I used while I was collecting data for the model.
5. data_manipulator: I used the code here to get some statistical information about the dataset with images for the model training and validation.

## How to Install and Run the Project
I you are an ordinary user, first download the piano_interface directory. If you have Python installed (version 3.9 or higher) then open a command line and navigate to the downloaded directory. After that run the following commands one by one:
```bash
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```
If your graphics card supports DirectX 12 it is highly recommended to install tensorflow-directml-plugin in order to have good performance. To do so just run
```bash
pip install tensorflow-directml-plugin
```
Now to launch the application just type main.py and hit Enter.

## How to Use the Project

