FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt  

# Copy the rest of the application files
COPY . .  

# Expose the port that Flask runs on
EXPOSE 5000  

# Command to run the application using Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
