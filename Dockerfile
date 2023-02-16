FROM python:3.8 # establishes the operating environment for the image
# copies all the files in the current directory into the image
COPY . ./   #installs necessary libraries
RUN pip3 install -r requirements.txt # installs the necessary libraries into the image
CMD ["assign2.py"] # list the command to run
ENTRYPOINT ["python"] # provides the starting run command to the image