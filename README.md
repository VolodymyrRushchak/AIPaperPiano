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
First, download the piano_interface directory. If you have Anaconda installed then open the command line and navigate to the downloaded directory. After that run the following commands one by one:
```bash
conda create --name ai_paper_piano python=3.9 
conda activate ai_paper_piano 
pip install -r requirements.txt
```
If your graphics card supports DirectX 12 it is highly recommended to install tensorflow-directml-plugin in order to have good performance. To do so just run
```bash
pip install tensorflow-directml-plugin
```
Now to launch the application just run
```bash
python main.py
```
! There is a chance that the model's weights haven't been downloaded properly. The size of the fine_tuned_model.data-00000-of-00001 file should be around 487MB. If it is smaller, then download the file [here](https://github.com/VolodymyrRushchak/AIPaperPiano/raw/main/piano_interface/assets/ai_model/fine_tuned_model.data-00000-of-00001?download=) and manually put it in the piano_interface\assets\ai_model folder.

## How to Use the Project
First, print the two halves of the piano templates. Then fix them on your table in the right order (e.g. using scotch tape). Then put your laptop behind the improvised piano and point your web camera on it, so that the whole piano can be seen on the screen. <br />
After that, using your mouse, click around the piano to form a green rectangle. It is important to start from the edges of the black keys like this:

<img src="https://github.com/VolodymyrRushchak/AIPaperPiano/assets/93164951/f9b75bbf-5fe7-4aee-bf5a-0ec835831d50" alt="Example Image" width="650" height="165">

After you do that, enjoy playing!
