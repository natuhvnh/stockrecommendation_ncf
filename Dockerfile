FROM ubuntu

# Install dependencies
RUN apt-get update
RUN apt-get install python3.6
RUN apt-get install -y python3-pip
# RUN apt-get update -y && \
#   apt-get install -y python3-pip python3-dev
RUN pip3 install --upgrade pip
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install Keras
RUN pip3 install tensorflow
RUN pip3 install flask
RUN pip3 install flask-restplus
RUN pip3 install Werkzeug==0.15.5
RUN pip3 install mysql-connector
RUN pip3 install sqlalchemy


# Create working directory and copy code
RUN mkdir /stockrecommendation_ncf
WORKDIR /stockrecommendation_ncf
COPY . /stockrecommendation_ncf
# Port variable
ENV PORT=8888
# ENTRYPOINT FLASK_APP=/stockrecommendation_ncf/docker_test.py # specify program run when container starts
# Run command
CMD python3 app.py
