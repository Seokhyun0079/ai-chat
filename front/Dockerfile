# Use an official Node.js runtime as a parent image
FROM node:20

# Set the working directory in the container
WORKDIR /front

# Copy the current directory contents into the container at /api
COPY . /front/

# Install any needed packages specified in package.json
RUN npm install

# Make port 8000 available to the world outside this container
EXPOSE 3000

# Run app.js when the container launches
CMD ["npm", "run", "dev"] 