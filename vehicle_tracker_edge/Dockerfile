FROM ultralytics/ultralytics:latest-cpu

# Copy the project folder
COPY . /home/vehical_counter

# Set environment variable to make apt-get non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies from packages.txt
RUN apt-get update && \
    apt-get install ntp -y && \
    echo "System packages installed successfully!"

# Install Python dependencies from requirements.txt
RUN pip install firebase_admin && \
    echo "Python packages installed successfully!"

# Set the timezone to Asia/Colombo
RUN ln -fs /usr/share/zoneinfo/Asia/Colombo /etc/localtime && \
    echo "Asia/Colombo" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    echo "Timezone set to Asia/Colombo!"

# Set the working directory
WORKDIR /home/vehical_counter

# Start the Python scripts and write logs in append mode
CMD python3 main.py




# docker build -t vehical_counting:bootcamp .
# docker run -dt --name jetson -v E:\BootCamp\week_4\Resources\shared_resorces_with_docker:/home/vehical_counter/configs  --restart always vehical_counting:bootcamp
# docker exec -it jetson /bin/bash
# docker logs -f jetson
# docker stop jetson

