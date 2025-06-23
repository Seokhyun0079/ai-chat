# Use an official Python runtime as a parent image
FROM python:3.10.7

# Set the working directory in the container
WORKDIR /chat-ai-api

# Copy the current directory contents into the container at /api
COPY . /chat-ai-api/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 8000

# Run api.py when the container launches
CMD ["python", "./api.py"]