## Author: Shaurya Chandhoke
### Edge Detection

# Instructions for the User Running the Program
## Prerequisites
#### Creating a Python 3 Virtual Environment
It's essential the user running the program have the following set up:
- A terminal/shell
- Python 3
- Python 3 pip
   - On Mac/Unix systems, installing Python 3 pip is as simple as `sudo apt-get install python3-pip`
   
After both are installed, it's **highly recommended** a python 3 virtual environment is created. To create one:

First install the `virtualenv` package via pip
```shell script
pip3 install virtualenv
```
After that's installed, create a python 3 virtual environment in the root directory of this project
```shell script
virtualenv venv
```
Once created, active the virtual environment
```shell script
source venv/bin/activate
```
If everything went smoothly, you should see a `(venv)` next to your terminal command line.

Now we can proceed with installing the prerequisite pip packages.

#### Installing the Prerequisite Packages
Included in the submission is a special *requirements.txt* file specially made for pip installations. In your terminal,
please run:
```shell script
pip3 install -r requirements.txt
```
It will install all the prerequisite python packages needed to run the program. You may open the file to view them.

## Running the Program
A quick way to run the program with its default configuration is:
```shell script
python3 edge_detector.py <image file>
```

However, I've included a way to allow the user to fine tune the program.
To see all options available for the user:
```shell script
python3 edge_detector.py --help
```

Below is a sample on how a user might run the program against an image:
```shell script
python3 edge_detector.py ./img/red.pgm --sigma 2 --threshold 60
```

Sometimes, you may not want to view the output, but simply save:
```shell script
python3 edge_detector.py ./img/plane.pgm --threshold 30 --quiet 
```

Other times, you may not want to save, but simply view the output:
```shell script
python3 edge_detector.py ./img/lenna.png --sigma 2 --threshold 55 --nosave
```

If you do not pass the `--nosave` flag, all images will be saved in the **./out**
directory

## The Folder Structure of the Project
Contained within the submission should be a series of files and folders:

- /img
   - A series of sample images to test the program from. You may use your own if you like as long as you include the 
   correct path to the image. The program will let you know if it cannot find the image file.
- /out
   - A directory the program utilizes to write it's output images to. Please do not delete this directory.
- /src
   - A directory containing the source code for the image processing. I placed the code in this directory for better 
   readability and segmentation.
- edge_detector.py
   - The Python file that contains the main function that will start the program. It is the file that will be run.
- requirements.txt
   - A special pip compatible package installation file that makes installation of prerequisite packages more 
   streamlined.
- README.md
   - An instructional file meant to serve as a quick How-To for running the program.